import { MelonyPlugin, Event } from "melony";
import { z } from "zod";
import * as fs from "fs";
import { Stagehand } from "@browserbasehq/stagehand";
import type { ModelMessage, V3Options } from "@browserbasehq/stagehand";
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
// Helpers
// ---------------------------------------------------------------------------

function normalizeAssistantMessage(message: ModelMessage): string[] {
  if (typeof message?.content === "string") {
    return [message.content];
  }
  if (Array.isArray(message?.content)) {
    return message.content
      .map((m: any) => {
        if (typeof m === "string") return m;
        if (m.type === "text") return m.text;
        if (m.type === "tool-call") {
          const tc = m;
          if (tc.toolName === "act") return `> **Action:** ${tc.input?.action}`;
          if (tc.toolName === "goto") return `> **Navigate to:** ${tc.input?.url}`;
          if (tc.toolName === "extract") return `> **Extracting information...**`;
          if (tc.toolName === "observe") return `> **Observing page...**`;
          return `> **Tool Call:** ${tc.toolName}`;
        }
        return null;
      })
      .filter((m): m is string => m !== null);
  }
  return [];
}

function normalizeToolMessage(message: ModelMessage): string {
  const content = message.content;
  if (!content) return "";

  // Try to parse if it's JSON
  let data;
  if (typeof content === "string") {
    try {
      data = JSON.parse(content);
    } catch {
      data = content;
    }
  } else {
    data = content;
  }

  // Handle Stagehand's tool-result array
  if (Array.isArray(data)) {
    return data
      .map((item: any) => {
        if (item.type === "tool-result") {
          const toolName = item.toolName;
          const outputValue = item.output?.value;
          const success = outputValue?.success ?? true;
          const emoji = success ? "✅" : "❌";

          let details = "";
          if (toolName === "goto") {
            details = `Reached ${outputValue?.url || "destination"}`;
          } else if (toolName === "act") {
            details = outputValue?.action || "Action successful";
          } else if (toolName === "extract") {
            details = "Data extracted";
            if (outputValue) {
              const summary = JSON.stringify(outputValue);
              details += ": " + (summary.length > 200 ? summary.slice(0, 200) + "..." : summary);
            }
          } else if (toolName === "done") {
            details = outputValue?.reasoning || "Task finished";
          } else if (toolName === "ariaTree") {
            // For ariaTree, item.output.value is often an array of content blocks
            if (Array.isArray(outputValue) && outputValue[0]?.text) {
              const text = outputValue[0].text;
              details = "Accessibility tree retrieved (" + (text.length > 100 ? text.slice(0, 100).replace(/\n/g, " ") + "..." : text.replace(/\n/g, " ")) + ")";
            } else {
              details = "Accessibility tree retrieved";
            }
          } else if (toolName === "keys") {
            details = `Pressed ${outputValue?.value || "keys"}`;
          } else {
            const summary = JSON.stringify(outputValue || {});
            details = summary.length > 200 ? summary.slice(0, 200) + "..." : summary;
          }

          return `  - *Result (${toolName}):* ${emoji} ${details}`;
        }
        return `  - *Result:* ${JSON.stringify(item).slice(0, 200)}`;
      })
      .join("\n");
  }

  const strContent = typeof content === "string" ? content : JSON.stringify(content);
  if (strContent.length > 500) {
    return `  - *Result:* ${strContent.slice(0, 500)}...`;
  }
  return `  - *Result:* ${strContent}`;
}

// ---------------------------------------------------------------------------
// Plugin factory
// ---------------------------------------------------------------------------

export const browserToolsPlugin = (
  options: BrowserToolsOptions = {}
): MelonyPlugin<any, any> => {
  const SESSION_KEY = "__browser_tools_plugin_session__";

  type BrowserSessionState = {
    stagehand?: Stagehand;
    initPromise?: Promise<Stagehand>;
    usage: {
      inputTokens: number;
      outputTokens: number;
      totalTokens: number;
    };
  };

  const getSession = (): BrowserSessionState => {
    const g = globalThis as typeof globalThis & {
      [SESSION_KEY]?: BrowserSessionState;
    };
    if (!g[SESSION_KEY]) {
      g[SESSION_KEY] = {
        usage: { inputTokens: 0, outputTokens: 0, totalTokens: 0 },
      };
    } else if (!g[SESSION_KEY].usage) {
      g[SESSION_KEY].usage = { inputTokens: 0, outputTokens: 0, totalTokens: 0 };
    }
    return g[SESSION_KEY]!;
  };

  const isSessionError = (error: unknown) => {
    const message = (error as Error | undefined)?.message?.toLowerCase() || "";
    return (
      message.includes("target closed") ||
      message.includes("has been closed") ||
      message.includes("browser has been closed") ||
      message.includes("context closed") ||
      message.includes("session closed")
    );
  };

  const clearSession = async () => {
    const session = getSession();
    const current = session.stagehand;
    session.stagehand = undefined;
    session.initPromise = undefined;
    if (current) {
      await current.close().catch(() => { });
    }
  };

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
    const session = getSession();

    if (session.stagehand) {
      try {
        // Probe active page to ensure the session is still valid.
        const page = session.stagehand.context.activePage();
        if (page) {
          return session.stagehand;
        }
      } catch {
        await clearSession();
      }
    }

    if (session.initPromise) {
      return session.initPromise;
    }

    session.initPromise = (async () => {
      if (options.userDataDir && !fs.existsSync(options.userDataDir)) {
        console.log(
          `[browser-tools] Creating userDataDir: ${options.userDataDir}`
        );
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

      const sh = new Stagehand(opts);
      await sh.init();
      session.stagehand = sh;
      return sh;
    })()
      .catch(async (error) => {
        await clearSession();
        throw error;
      })
      .finally(() => {
        session.initPromise = undefined;
      });

    return session.initPromise;
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

    function actionResult(
      action: string,
      toolCallId: string,
      result: string
    ) {
      return {
        type: "action:result",
        data: { action, toolCallId, result },
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

        const sh = await ensureStagehand();

        const agent = sh.agent({
          mode: "dom",
          ...modelConfig,
          systemPrompt: options.systemPrompt || "You are a helpful browser automation assistant. Achieve the user's goal by navigating, interacting with elements, and extracting information as needed.",
          stream: true,
        });

        const streamResult = await agent.execute({
          instruction,
          maxSteps: 20,
        });

        for await (const part of streamResult.fullStream) {
          // if (part.type === "text-delta" || part.type === "reasoning-delta") {
          //   const delta = (part as any).textDelta || (part as any).reasoningDelta;
          //   if (delta) {
          //     yield {
          //       type: "browser:status",
          //       data: { message: delta },
          //     } as BrowserStatusEvent;
          //   }
          // }

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

        const executionTrace: string[] = [];
        if (result.messages) {
          for (const m of result.messages) {
            if (m.role === "assistant") {
              executionTrace.push(...normalizeAssistantMessage(m));
            } else if (m.role === "tool") {
              executionTrace.push(normalizeToolMessage(m));
            }
          }
        }

        console.log(
          "executionTrace",
          JSON.stringify(executionTrace, null, 2)
        );

        // Yield usage
        if (result.usage) {
          const state = getSession();
          const usageEventType = "usage:update";
          const usageScope = "browser_action";
          const modelId =
            typeof modelConfig.model === "string"
              ? modelConfig.model
              : (modelConfig.model as any)?.modelName || "openai/gpt-4o";

          const turn = {
            inputTokens: result.usage.input_tokens ?? 0,
            outputTokens: result.usage.output_tokens ?? 0,
            totalTokens:
              (result.usage.input_tokens ?? 0) +
              (result.usage.output_tokens ?? 0),
          };
          state.usage.inputTokens += turn.inputTokens;
          state.usage.outputTokens += turn.outputTokens;
          state.usage.totalTokens += turn.totalTokens;

          yield {
            type: usageEventType,
            data: {
              scope: usageScope,
              model: modelId,
              turn,
              session: {
                inputTokens: state.usage.inputTokens,
                outputTokens: state.usage.outputTokens,
                totalTokens: state.usage.totalTokens,
              },
            },
          } as Event;
        }

        // Final state update
        yield* yieldState(sh);

        yield {
          type: "browser:status",
          data: { message: result.message },
        } as BrowserStatusEvent;

        const history = executionTrace.length > 0
          ? "\n\n### Browser Execution History\n\n" + executionTrace.join("\n")
          : "";

        yield actionResult("browser_action", toolCallId, result.message + history);
      } catch (error: any) {
        if (isSessionError(error)) {
          await clearSession();
        }
        yield actionResult("browser_action", toolCallId, `Error: ${error.message}`);
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
        yield actionResult("browser_screenshot", toolCallId, `URL: ${url}\nTitle: ${title}`);
      } catch (error: any) {
        if (isSessionError(error)) {
          await clearSession();
        }
        yield actionResult("browser_screenshot", toolCallId, `Error: ${error.message}`);
      }
    });

    // -- browser_cleanup ----------------------------------------------------

    builder.on("action:browser_cleanup" as any, async function* (event) {
      const { toolCallId } = event.data;
      try {
        await clearSession();
        yield actionResult("browser_cleanup", toolCallId, "Browser closed");
      } catch (error: any) {
        await clearSession();
        yield actionResult("browser_cleanup", toolCallId, `Error: ${error.message}`);
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
        yield ui.event(ui.text(event.data.message, { size: "xs", color: event.data.severity === "error" ? "destructiveForeground" : "foreground" }));
      }
    );

    builder.on(
      "browser:state-update" as any,
      async function* (event: BrowserStateUpdateEvent) {
        if (event.data.screenshot) {
          yield ui.event(
            ui.box({ border: true, padding: "md", radius: "md" }, [
              ui.col({ gap: "md" }, [
                ui.col({ gap: "xs" }, [
                  ui.text(event.data.title, { size: "sm" }),
                  ui.text(event.data.url, { size: "xs", color: "mutedForeground" }),
                ]),
                ui.box({ border: true, radius: "md", overflow: "hidden" }, [
                  ui.image(`data:image/jpeg;base64,${event.data.screenshot}`),
                ]),
              ]),
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
