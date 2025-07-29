────────────────────────────────────────────────────────
Agent‑to‑Agent Protocol (A2A) — Comprehensive Reference
Spec version : 0.2.3   •   Publish date : 2025‑06‑14
Homepage     : https://a2aproject.github.io/A2A/latest/
────────────────────────────────────────────────────────

1. PURPOSE & SCOPE
────────────────────────────────────────────────────────
A2A defines **how independent AI agents communicate as peers**, with
built‑in discovery, capability negotiation, long‑running task management,
streaming, push notifications and enterprise‑grade security. Its goals are:

• Interoperability between heterogeneous agent stacks  
• Dynamic discovery of skills & auth requirements  
• Secure, opaque collaboration (no internal state leaks)  
• First‑class support for async / streaming & human‑in‑loop workflows  
• Basing everything on well‑known Web standards (HTTP, JSON‑RPC 2.0, SSE) :contentReference[oaicite:0]{index=0}

2. CORE MODEL
────────────────────────────────────────────────────────
Concept              | Description
-------------------- | -----------------------------------------------------
A2A Client           | Initiator acting for a user / upstream agent
A2A Server           | Remote agent exposing an A2A endpoint
Task                 | Long‑lived stateful unit of work
Message              | One conversational turn; contains **Part[]**
Artifact             | Durable output produced by a task
Agent Card           | JSON metadata document describing identity, skills,
                     | endpoint URL & security requirements :contentReference[oaicite:1]{index=1}

3. TRANSPORT & ENCODING
────────────────────────────────────────────────────────
Transport            : **HTTPS** (TLS 1.2+) POST to Agent Card .url  
Payload format       : **JSON‑RPC 2.0** (`Content‑Type: application/json`)  
Streaming channel    : **Server‑Sent Events** (`text/event-stream`) used by  
                      `message/stream` and `tasks/resubscribe`.  
SSE payload          : each event’s *data* field is a full JSON‑RPC Response. :contentReference[oaicite:2]{index=2}

4. SECURITY MODEL
────────────────────────────────────────────────────────
Layered security = transport TLS + declared auth schemes.

• Auth schemes are declared in `AgentCard.securitySchemes`  
  – mirrors OpenAPI: `apiKey`, `http` (Basic / Bearer), `oauth2`,
    `openIdConnect`.  
• Required scheme combinations are listed in `AgentCard.security`  
• Missing / invalid credentials → HTTP 401/403 with `WWW‑Authenticate`.  
• Optional **secondary (in‑task) auth** — task transitions to
  `auth-required` until satisfied. :contentReference[oaicite:3]{index=3}

5. DISCOVERY  –  THE AGENT CARD
────────────────────────────────────────────────────────
Recommended URL      : `https://{host}/.well‑known/agent.json`  
Minimal top‑level schema (TS‑style):

```ts
interface AgentCard {
  name: string;
  description: string;
  url: string;                 // base A2A endpoint
  provider?: AgentProvider;    // org info
  version: string;             // agent impl version
  iconUrl?: string;            // 64×64+ PNG/SVG
  documentationUrl?: string;
  capabilities: AgentCapabilities;
  securitySchemes?: { [name: string]: SecurityScheme };
  security?: { [name: string]: string[] }[];
  defaultInputModes: string[];   // e.g. ["text/plain","application/json"]
  defaultOutputModes: string[];  // e.g. ["application/json"]
  skills: AgentSkill[];
  supportsAuthenticatedExtendedCard?: boolean;  // if true: expose
}
``` :contentReference[oaicite:4]{index=4}

Key sub‑objects
• **AgentCapabilities** → `{ streaming?: bool, pushNotifications?: bool,
  stateTransitionHistory?: bool, extensions?: AgentExtension[] }`  
• **AgentSkill** → `{ id, name, description, tags[], examples?[],
  inputModes?, outputModes? }`  
• **SecurityScheme** → union of OpenAPI schemes (API Key, HTTP, OAuth2,
  OpenID Connect). :contentReference[oaicite:5]{index=5}

6. DATA OBJECTS
────────────────────────────────────────────────────────
6.1 Task
```ts
interface Task {
  id: string;
  contextId: string;
  status: TaskStatus;
  artifacts?: Artifact[];
  history?: Message[];
  metadata?: Record<string,any>;
  kind: "task";
}
``` :contentReference[oaicite:6]{index=6}

6.2 TaskStatus & TaskState enum  
States: `submitted | working | input‑required | completed | canceled |
failed | rejected | auth‑required | unknown`.  
`completed, canceled, failed, rejected` are **terminal**. :contentReference[oaicite:7]{index=7}

6.3 Message & Part
```ts
type Part = TextPart | FilePart | DataPart;

interface Message {
  role: "user" | "agent";
  parts: Part[];
  messageId: string;
  contextId?: string;
  taskId?: string;
  referenceTaskIds?: string[];
  metadata?: Record<string,any>;
  extensions?: string[];
  kind: "message";
}
``` :contentReference[oaicite:8]{index=8}

• **TextPart**→ `{ kind:"text", text }`  
• **FilePart**→ `{ kind:"file", file: FileWithBytes | FileWithUri }`  
• **DataPart**→ `{ kind:"data", data: {...} }`  
• **Artifact**→ `{ artifactId, parts:Part[], name?, description?, metadata?, extensions? }` :contentReference[oaicite:9]{index=9}

6.4 PushNotificationConfig  
```ts
interface PushNotificationConfig {
  id?: string;           // server‑assigned
  url: string;           // HTTPS webhook
  token?: string;        // opaque HMAC/shared secret
  authentication?: PushNotificationAuthenticationInfo;
}
``` :contentReference[oaicite:10]{index=10}

6.5 JSON‑RPC Structures  
`JSONRPCRequest`, `JSONRPCResponse`, `JSONRPCError` – standard 2.0
shapes, plus A2A‑specific `result` payloads. :contentReference[oaicite:11]{index=11}

7. RPC METHOD CATALOG
────────────────────────────────────────────────────────
All calls are **POST** `AgentCard.url` with a JSON‑RPC 2.0 request.

Method                       | Params → Result (success)                      | Notes
---------------------------- | --------------------------------------------- | ------------------------------------------------------
`message/send`               | **MessageSendParams** → `Task | Message`      | Sync request; client polls `tasks/get` or SSE
`message/stream`             | MessageSendParams → *SSE* stream of `Message` \| `Task` \| `TaskStatusUpdateEvent` \| `TaskArtifactUpdateEvent` | Requires `capabilities.streaming=true`
`tasks/get`                  | **/* TaskQueryParams** → `Task`                  | Poll task status / history
`tasks/cancel`               | **TaskIdParams** → `Task`                     | Attempt cancellation
`tasks/pushNotificationConfig/set` | **TaskPushNotificationConfig** → same  | Requires `capabilities.pushNotifications=true`
`tasks/pushNotificationConfig/get` | GetTaskPushNotificationConfigParams → TaskPushNotificationConfig |
`tasks/pushNotificationConfig/list`| ListTaskPushNotificationConfigParams → TaskPushNotificationConfig[] |
`tasks/pushNotificationConfig/delete`| TaskIdParams + configId → (void)      |
`tasks/resubscribe`          | TaskIdParams → *SSE*                          | Re‑attach after network drop
`agent/authenticatedExtendedCard`  | AuthenticatedExtendedCardParams → AgentCard | Returns richer card after auth :contentReference[oaicite:12]{index=12}

Essential parameter types  
```ts
interface MessageSendParams {
  message: Message;
  configuration?: {
    acceptedOutputModes: string[];
    historyLength?: number;
    pushNotificationConfig?: PushNotificationConfig;
    blocking?: boolean;        // if true: HTTP held open until terminal
  };
  metadata?: Record<string,any>;
}

interface TaskQueryParams   { id: string; historyLength?: number }
interface TaskIdParams      { id: string }
interface TaskPushNotificationConfig { taskId: string; pushNotificationConfig: PushNotificationConfig }
``` :contentReference[oaicite:13]{index=13}

7.1 Streaming events  
• **TaskStatusUpdateEvent** → `{ taskId, contextId, kind:"status‑update", status:TaskStatus, final:boolean }`  
• **TaskArtifactUpdateEvent** → `{ taskId, contextId, kind:"artifact-update", artifact:Artifact, append?, lastChunk? }` :contentReference[oaicite:14]{index=14}

8. ERROR HANDLING
────────────────────────────────────────────────────────
A2A reuses JSON‑RPC codes and reserves **‑32000…‑32099** for protocol‑specific
errors.

Code   | Name                              | Typical meaning
------ | --------------------------------- | ----------------------------------------
‑32001 | TaskNotFoundError                 | Unknown/expired task ID
‑32002 | TaskNotCancelableError            | Task is not in cancelable state
‑32003 | PushNotificationNotSupportedError | Server lacks push‑notification capability
‑32004 | UnsupportedOperationError         | Feature or parameter not supported
‑32005 | ContentTypeNotSupportedError      | Unaccepted MIME type in parts/artifacts
‑32006 | InvalidAgentResponseError         | Server produced invalid response shape :contentReference[oaicite:15]{index=15}  
Standard codes ‑32700…‑32603 follow JSON‑RPC 2.0. Include explanatory
`message` and optional structured `data`.

9. TYPICAL WORKFLOWS (NON‑NORMATIVE)
────────────────────────────────────────────────────────
**Synchronous / Polling**

1. Fetch Agent Card → pick skill → form first `Message`  
2. `message/send` → server returns `Task { state:working }`  
3. Client polls `tasks/get` until terminal state  
4. On `completed` → download artifacts/messages as needed

**Streaming**

1. `message/stream` → HTTP 200 + SSE  
2. Parse stream events; optionally `tasks/resubscribe` if dropped  
3. Close stream when final status update or artifact chunk with `lastChunk:true`

**Input‑Required turn**

*Task* enters `input‑required` → client collects user input → call
`message/send` with same `taskId` (inside Message.taskId). Task continues.

**Push‑notifications**

• Register via `tasks/pushNotificationConfig/set` → Server POSTs JSON‑RPC
envelopes to `url`, signed per `token` and `authentication`. :contentReference[oaicite:16]{index=16}

10. EXTENSIBILITY
────────────────────────────────────────────────────────
• **Extensions** declared in `AgentCapabilities.extensions` and echoed in
messages/artifacts via the `extensions` string array.  
• New RPC methods MAY be added using `namespace/verb` naming, provided
clients fail gracefully on `‑32601 Method not found`.  
• New TaskState values SHOULD reserve lower‑case, dash‑separated strings and
be treated as non‑terminal unless explicitly documented.

────────────────────────────────────────────────────────
END OF SPEC (A2A v0.2.3)
──────────────────────────────────────────────────────── */

