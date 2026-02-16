import { MelonyPlugin, Event } from "melony";
import { z } from "zod";
import * as fs from "fs";
import { Stagehand } from "@browserbasehq/stagehand";
import type { V3Options } from "@browserbasehq/stagehand";
import { ui } from "@melony/ui-kit/server";

// ---------------------------------------------------------------------------
// Tool definitions
// ---------------------------------------------------------------------------

export const browserToolDefinitions = {
  browser_action: {
    description:
      "Perform a multi-step browser task using natural language. The agent will autonomously navigate, interact, and extract data to fulfill the instruction.",
    inputSchema: z.object({
      instruction: z
        .string()
        .describe(
          "The high-level goal to achieve, e.g. 'Go to GitHub, find the most starred repo for typescript, and tell me its name and stars count'"
        ),
    }),
  },
  browser_screenshot: {
    description: "Take a screenshot of the current page and return page info.",
    inputSchema: z.object({}),
  },
  browser_cleanup: {
    description: "Close the browser and release all resources.",
    inputSchema: z.object({}),
  },
};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface BrowserToolsOptions {
  /**
   * Model configuration. Accepts:
   * - A string like "openai/gpt-4o", "anthropic/claude-3-5-sonnet-latest", or just "gpt-4o" / "claude-3-5-sonnet-latest"
   * - A Stagehand model config object: { modelName: "gpt-4o", apiKey?: "..." }
   */
  model?: V3Options["model"];
  /** Additional Stagehand constructor options */
  stagehandConfig?: Partial<Omit<V3Options, "env">>;

  /** The directory to store the browser's user data */
  userDataDir?: string;

  // system prompt
  systemPrompt?: string;
}

export interface BrowserStatusEvent extends Event {
  type: "browser:status";
  data: { message: string; severity?: "info" | "success" | "error" };
}

export interface BrowserStateUpdateEvent extends Event {
  type: "browser:state-update";
  data: {
    url: string;
    title: string;
    screenshot?: string;
  };
}

// ---------------------------------------------------------------------------
// Plugin factory
// ---------------------------------------------------------------------------

export const browserToolsPlugin = (
  options: BrowserToolsOptions = {}
): MelonyPlugin<any, any> => {
  let stagehand: Stagehand | undefined;

  const resolveModelConfig = () => {
    const model = options.model;

    if (!model) {
      return { model: "openai/gpt-4o" };
    }

    if (typeof model === "string") {
      return { model };
    }

    if (
      typeof model === "object" &&
      "specificationVersion" in model &&
      (model as any).specificationVersion === "v3"
    ) {
      const v3Model = model as any;
      const provider = v3Model.config?.provider?.split(".")[0];
      const modelId = v3Model.modelId;
      if (provider && modelId) {
        return { model: `${provider}/${modelId}` };
      }
    }

    return { model };
  };

  async function ensureStagehand(): Promise<Stagehand> {
    if (stagehand) return stagehand;

    if (options.userDataDir && !fs.existsSync(options.userDataDir)) {
      console.log(`[browser-tools] Creating userDataDir: ${options.userDataDir}`);
      fs.mkdirSync(options.userDataDir, { recursive: true });
    }

    const opts: V3Options = {
      env: "LOCAL",
      verbose: 1,
      selfHeal: true,
      experimental: true,
      disableAPI: true,
      ...options.stagehandConfig,
      localBrowserLaunchOptions: {
        ...options.stagehandConfig?.localBrowserLaunchOptions,
        ...(options.userDataDir ? { userDataDir: options.userDataDir } : {}),
      },
    };

    stagehand = new Stagehand(opts);
    await stagehand.init();
    return stagehand;
  }

  /** Get the active page from the stagehand context, throwing if none */
  function getPage(sh: Stagehand) {
    const page = sh.context.activePage();
    if (!page) {
      throw new Error("No active browser page. Navigate to a URL first.");
    }
    return page;
  }

  return (builder) => {
    // -- helpers ------------------------------------------------------------

    async function* yieldState(sh: Stagehand) {
      try {
        const page = getPage(sh);
        if (!page) return;

        // Wait for page to stabilize before screenshotting
        await page.waitForLoadState("load", 5000).catch(() => { });
        await page.waitForLoadState("networkidle", 2000).catch(() => { });
        await page.waitForTimeout(500).catch(() => { });

        const url = page.url();
        const title = await page.title();
        const buf = await page
          .screenshot({ type: "jpeg", quality: 60 })
          .catch(() => null);
        const base64 = buf ? Buffer.from(buf).toString("base64") : undefined;

        yield {
          type: "browser:state-update",
          data: { url, title, screenshot: base64 },
        } as BrowserStateUpdateEvent;
      } catch (e) {
        console.error("[browser-tools] state update failed:", e);
      }
    }

    function taskResult(
      action: string,
      toolCallId: string,
      data: Record<string, unknown>
    ) {
      return {
        type: "action:taskResult",
        data: { action, toolCallId, result: data },
      };
    }

    // -- browser_action ----------------------------------------------------

    builder.on("action:browser_action" as any, async function* (event) {
      const { toolCallId, instruction } = event.data;

      yield {
        type: "browser:status",
        data: { message: `Executing browser action: ${instruction}` },
      } as BrowserStatusEvent;

      try {
        const modelConfig = resolveModelConfig();

        console.log("MODEL CONFIG:", JSON.stringify(modelConfig, null, 2));

        const sh = await ensureStagehand();

        const agent = sh.agent({
          mode: "hybrid",
          ...modelConfig,
          systemPrompt: options.systemPrompt || "You are a helpful browser automation assistant. Achieve the user's goal by navigating, interacting with elements, and extracting information as needed.   Make responses concise and to the point.",
          stream: true,
        });

        const streamResult = await agent.execute({
          instruction,
          maxSteps: 20,
        });

        for await (const part of streamResult.fullStream) {
          if (part.type === "text-delta" || part.type === "reasoning-delta") {
            const delta = (part as any).textDelta || (part as any).reasoningDelta;
            if (delta) {
              yield {
                type: "browser:status",
                data: { message: delta },
              } as BrowserStatusEvent;
            }
          }

          if (part.type === "error") {
            yield {
              type: "browser:status",
              data: {
                message: `Error: ${(part as any).error}`,
                severity: "error",
              },
            } as BrowserStatusEvent;
          }

          if (part.type === "tool-call") {
            const tc = part as any;
            let msg = `Action: ${tc.toolName}`;
            if (tc.toolName === "act")
              msg = `Browser action: ${tc.input?.action}`;
            else if (tc.toolName === "goto")
              msg = `Navigating to: ${tc.input?.url}`;
            else if (tc.toolName === "extract")
              msg = `Extracting information...`;

            yield {
              type: "browser:status",
              data: { message: msg },
            } as BrowserStatusEvent;
          }

          if (part.type === "tool-result") {
            // yield* yieldState(sh);
          }
        }

        const result = await streamResult.result;

        // Final state update
        yield* yieldState(sh);

        yield taskResult("browser_action", toolCallId, {
          success: result.success,
          message: result.message,
        });
      } catch (error: any) {
        yield taskResult("browser_action", toolCallId, {
          success: false,
          error: error.message,
        });
      }
    });

    // -- browser_screenshot -------------------------------------------------

    builder.on("action:browser_screenshot" as any, async function* (event) {
      const { toolCallId } = event.data;
      try {
        const sh = await ensureStagehand();
        const page = getPage(sh);
        const url = page.url();
        const title = await page.title();

        yield* yieldState(sh);
        yield taskResult("browser_screenshot", toolCallId, {
          success: true,
          url,
          title,
        });
      } catch (error: any) {
        yield taskResult("browser_screenshot", toolCallId, {
          success: false,
          error: error.message,
        });
      }
    });

    // -- browser_cleanup ----------------------------------------------------

    builder.on("action:browser_cleanup" as any, async function* (event) {
      const { toolCallId } = event.data;
      try {
        if (stagehand) {
          await stagehand.close();
          stagehand = undefined;
        }
        yield taskResult("browser_cleanup", toolCallId, {
          success: true,
          message: "Browser closed",
        });
      } catch (error: any) {
        stagehand = undefined;
        yield taskResult("browser_cleanup", toolCallId, {
          success: false,
          error: error.message,
        });
      }
    });

    // Register UI handlers
    browserToolsUIPlugin()(builder);
  };
};

// ---------------------------------------------------------------------------
// UI Plugin
// ---------------------------------------------------------------------------

export const browserToolsUIPlugin =
  (): MelonyPlugin<any, any> => (builder) => {
    builder.on(
      "browser:status" as any,
      async function* (event: BrowserStatusEvent) {
        yield ui.event(ui.status(event.data.message, event.data.severity));
      }
    );

    builder.on(
      "browser:state-update" as any,
      async function* (event: BrowserStateUpdateEvent) {
        if (event.data.screenshot) {
          yield ui.event(
            ui.resourceCard(event.data.title, event.data.url, [
              ui.image(`data:image/jpeg;base64,${event.data.screenshot}`),
            ])
          );
        }
      }
    );
  };

// ---------------------------------------------------------------------------
// Plugin Entry for Registry
// ---------------------------------------------------------------------------

export const plugin = {
  name: "browser-tools",
  description: "Browse the web and interact with pages using Stagehand",
  toolDefinitions: browserToolDefinitions,
  factory: (options: BrowserToolsOptions) => browserToolsPlugin(options),
};

export default plugin;
