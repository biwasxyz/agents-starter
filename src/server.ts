import { createWorkersAI } from "workers-ai-provider";
import { callable, routeAgentRequest, type Schedule } from "agents";
import { getSchedulePrompt, scheduleSchema } from "agents/schedule";
import { AIChatAgent, type OnChatMessageOptions } from "@cloudflare/ai-chat";
import {
  convertToModelMessages,
  pruneMessages,
  stepCountIs,
  streamText,
  tool,
  type ModelMessage
} from "ai";
import { z } from "zod";

/**
 * The AI SDK's downloadAssets step runs `new URL(data)` on every file
 * part's string data. Data URIs parse as valid URLs, so it tries to
 * HTTP-fetch them and fails. Decode to Uint8Array so the SDK treats
 * them as inline data instead.
 */
function inlineDataUrls(messages: ModelMessage[]): ModelMessage[] {
  return messages.map((msg) => {
    if (msg.role !== "user" || typeof msg.content === "string") return msg;
    return {
      ...msg,
      content: msg.content.map((part) => {
        if (part.type !== "file" || typeof part.data !== "string") return part;
        const match = part.data.match(/^data:([^;]+);base64,(.+)$/);
        if (!match) return part;
        const bytes = Uint8Array.from(atob(match[2]), (c) => c.charCodeAt(0));
        return { ...part, data: bytes, mediaType: match[1] };
      })
    };
  });
}

export class ChatAgent extends AIChatAgent<Env> {
  maxPersistedMessages = 100;

  onStart() {
    // Configure OAuth popup behavior for MCP servers that require authentication
    this.mcp.configureOAuthCallback({
      customHandler: (result) => {
        if (result.authSuccess) {
          return new Response("<script>window.close();</script>", {
            headers: { "content-type": "text/html" },
            status: 200
          });
        }
        return new Response(
          `Authentication Failed: ${result.authError || "Unknown error"}`,
          { headers: { "content-type": "text/plain" }, status: 400 }
        );
      }
    });
  }

  @callable()
  async addServer(name: string, url: string) {
    return await this.addMcpServer(name, url);
  }

  @callable()
  async removeServer(serverId: string) {
    await this.removeMcpServer(serverId);
  }

  async onChatMessage(_onFinish: unknown, options?: OnChatMessageOptions) {
    const mcpTools = this.mcp.getAITools();
    const workersai = createWorkersAI({ binding: this.env.AI });

    const result = streamText({
      model: workersai("@cf/openai/gpt-oss-20b", {
        sessionAffinity: this.sessionAffinity
      }),
      system: `You are biwas.ai — Biwas Bhandari's personal AI agent. You speak on Biwas's behalf to visitors of his site. Be warm, direct, and concise. Mirror Biwas's voice: matter-of-fact, no hype, no marketing fluff. Short paragraphs. Plain language.

# How to respond
Answer the user directly in natural language. Tools are optional helpers — only call one when it's clearly needed (weather, math, scheduling a reminder, image analysis). For everything else (especially questions about Biwas), answer from the information below. **Never refuse a message just because no tool fits — always reply conversationally.**

# Identity
- Name: Biwas Bhandari
- Tagline: Building AI agents and automations with LangChain and LangGraph.
- Site: https://biwas.xyz
- Email: biwas2059@gmail.com
- Socials: GitHub, Twitter, LinkedIn (linked from biwas.xyz)

# About
Started coding in 2023. Self-taught through YouTube, freeCodeCamp, and building projects until things clicked. Now does freelance work — mostly full-stack web apps and AI integrations. Focuses on problems where AI genuinely helps rather than adds complexity. Works async and communicates clearly.

Can help with:
- Web apps from scratch (Next.js + FastAPI)
- Adding AI to existing products
- Deployment and infrastructure

Certification: Scientific Computing with Python (freeCodeCamp).

# Tech stack
React · Next.js · TypeScript · Python · LangChain · LangGraph · FastAPI · Postgres · Docker · Vercel · Cloudflare Workers

# Currently working on
- aibtc-mcp-server
- x402 endpoints on Stacks
- one-script openclaw setup

# Selected projects
- AIBTC MCP Server (2025) — MCP server for AI agents to interact with the Stacks blockchain: wallet management, token transfers, DeFi. Stack: TypeScript, MCP, Stacks.
- x402 Endpoints (2025) — Pay-per-call API endpoints on Stacks, built with the x402 protocol. Stack: x402, Stacks, Hono.
- AIBTC (2024) — Console for decentralized orgs: proposals, voting, AI agent execution. Stack: Next.js, LangChain, FastAPI.
- Asian Hiking Team (2025) — Travel agency website with a CMS for managing tours and content. Stack: Next.js, Strapi. With Dawa Sherpa.
- Jholpattey (2024) — Restaurant website. Managed hosting, domain, and deployment. Stack: Next.js.
- BTC Smart Wallet (2024) — AI-powered wallet with token swap suggestions. Stack: LangChain, stacks.js. With Salin Kattel.

# Experience
- 2024 → present — Open source contributor at aibtc.dev. Contributing to AI + Bitcoin tooling and infrastructure.
- 2024 — AI Integrator at Builders Academy. Built a smart contract analyzer and Clarity code writer; both submitted as bounties to aibtcdev.
- 2024 — Contract developer at Startino. Built an AI health assistant and migrated it from Python to TypeScript. Worked on AI marketing tools and revived a legacy React + Express project.
- 2023 — Frontend & Blockchain dev at Builders Academy. Worked on the Ordinals ecosystem. Built a P&L calculator based on buy/sell prices with 1099 form export.
- 2023 — Self-taught at freeCodeCamp.

# Hiring / contact
If a visitor wants to hire Biwas or get in touch, point them to biwas2059@gmail.com or biwas.xyz. Biwas is open to freelance work.

# Behavior rules
- Speak about Biwas in first person ("I") when natural for a personal agent, or third person ("Biwas") when clearer. Pick whichever reads better; don't switch mid-answer.
- Don't fabricate projects, dates, employers, or contact details. If asked something not in this prompt, say you're not sure and point them to biwas.xyz.
- Keep answers tight. A two-sentence answer beats a five-paragraph one for most questions.

${getSchedulePrompt({ date: new Date() })}

If the user asks to schedule a task, use the schedule tool.`,
      // Prune old tool calls to save tokens on long conversations
      messages: pruneMessages({
        messages: inlineDataUrls(await convertToModelMessages(this.messages)),
        toolCalls: "before-last-2-messages"
      }),
      tools: {
        // MCP tools from connected servers
        ...mcpTools,

        // Server-side tool: runs automatically on the server
        getWeather: tool({
          description: "Get the current weather for a city",
          inputSchema: z.object({
            city: z.string().describe("City name")
          }),
          execute: async ({ city }) => {
            // Replace with a real weather API in production
            const conditions = ["sunny", "cloudy", "rainy", "snowy"];
            const temp = Math.floor(Math.random() * 30) + 5;
            return {
              city,
              temperature: temp,
              condition:
                conditions[Math.floor(Math.random() * conditions.length)],
              unit: "celsius"
            };
          }
        }),

        // Client-side tool: no execute function — the browser handles it
        getUserTimezone: tool({
          description:
            "Get the user's timezone from their browser. Use this when you need to know the user's local time.",
          inputSchema: z.object({})
        }),

        // Approval tool: requires user confirmation before executing
        calculate: tool({
          description:
            "Perform a math calculation with two numbers. Requires user approval for large numbers.",
          inputSchema: z.object({
            a: z.number().describe("First number"),
            b: z.number().describe("Second number"),
            operator: z
              .enum(["+", "-", "*", "/", "%"])
              .describe("Arithmetic operator")
          }),
          needsApproval: async ({ a, b }) =>
            Math.abs(a) > 1000 || Math.abs(b) > 1000,
          execute: async ({ a, b, operator }) => {
            const ops: Record<string, (x: number, y: number) => number> = {
              "+": (x, y) => x + y,
              "-": (x, y) => x - y,
              "*": (x, y) => x * y,
              "/": (x, y) => x / y,
              "%": (x, y) => x % y
            };
            if (operator === "/" && b === 0) {
              return { error: "Division by zero" };
            }
            return {
              expression: `${a} ${operator} ${b}`,
              result: ops[operator](a, b)
            };
          }
        }),

        scheduleTask: tool({
          description:
            "Schedule a task to be executed at a later time. Use this when the user asks to be reminded or wants something done later.",
          inputSchema: scheduleSchema,
          execute: async ({ when, description }) => {
            if (when.type === "no-schedule") {
              return "Not a valid schedule input";
            }
            const input =
              when.type === "scheduled"
                ? when.date
                : when.type === "delayed"
                  ? when.delayInSeconds
                  : when.type === "cron"
                    ? when.cron
                    : null;
            if (!input) return "Invalid schedule type";
            try {
              this.schedule(input, "executeTask", description, {
                idempotent: true
              });
              return `Task scheduled: "${description}" (${when.type}: ${input})`;
            } catch (error) {
              return `Error scheduling task: ${error}`;
            }
          }
        }),

        getScheduledTasks: tool({
          description: "List all tasks that have been scheduled",
          inputSchema: z.object({}),
          execute: async () => {
            const tasks = this.getSchedules();
            return tasks.length > 0 ? tasks : "No scheduled tasks found.";
          }
        }),

        cancelScheduledTask: tool({
          description: "Cancel a scheduled task by its ID",
          inputSchema: z.object({
            taskId: z.string().describe("The ID of the task to cancel")
          }),
          execute: async ({ taskId }) => {
            try {
              this.cancelSchedule(taskId);
              return `Task ${taskId} cancelled.`;
            } catch (error) {
              return `Error cancelling task: ${error}`;
            }
          }
        })
      },
      stopWhen: stepCountIs(5),
      abortSignal: options?.abortSignal
    });

    return result.toUIMessageStreamResponse();
  }

  async executeTask(description: string, _task: Schedule<string>) {
    // Do the actual work here (send email, call API, etc.)
    console.log(`Executing scheduled task: ${description}`);

    // Notify connected clients via a broadcast event.
    // We use broadcast() instead of saveMessages() to avoid injecting
    // into chat history — that would cause the AI to see the notification
    // as new context and potentially loop.
    this.broadcast(
      JSON.stringify({
        type: "scheduled-task",
        description,
        timestamp: new Date().toISOString()
      })
    );
  }
}

export default {
  async fetch(request: Request, env: Env) {
    return (
      (await routeAgentRequest(request, env)) ||
      new Response("Not found", { status: 404 })
    );
  }
} satisfies ExportedHandler<Env>;
