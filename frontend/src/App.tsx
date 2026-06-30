import * as React from "react"
import ReactMarkdown from "react-markdown"
import remarkGfm from "remark-gfm"
import {
  BookOpen,
  Bot,
  CheckCircle2,
  Cpu,
  Database,
  ImagePlus,
  LoaderCircle,
  PanelRight,
  Radio,
  RotateCcw,
  Send,
  Square,
  Wrench,
  X,
} from "lucide-react"

import { Button } from "@/components/ui/button"
import { useTheme } from "@/components/theme-provider"
import { cn } from "@/lib/utils"

type AskEvent = [string, string, Diagnostics, string]

type Diagnostics = {
  status?: string
  stage?: "prepare" | "retrieve" | "generate" | "done" | string
  intent?: string
  timings_ms?: Record<string, number>
  draft?: {
    visible?: boolean
    chars?: number
    first_visible_ms?: number
  }
  retrieval?: {
    vector_index_backend?: string
    reranker_cache_hit?: boolean
    reranker_wait_ms?: number
    reranker_used?: boolean
    reranker_mode?: string
    top_k?: number
    candidate_k?: number
    configured_top_k?: number
    configured_candidate_k?: number
    reranker_candidate_count?: number
    retrieval_budget?: {
      mode?: string
      reason?: string
      top_k?: number
      candidate_k?: number
    }
    timings_ms?: Record<string, number>
  }
  top_sources?: Array<{
    id?: string
    title?: string
    source?: string
    snippet?: string
  }>
  agent?: {
    status?: string
    used?: boolean
    streamed?: boolean
    stream_chars?: number
    first_token_ms?: number
    wait_ms?: number
  }
  agent_used?: boolean
  agent_first_token_ms?: number
}

type StepKey = "prepare" | "retrieve" | "generate" | "done"
type ChatRole = "user" | "assistant"

type ChatMessage = {
  id: string
  role: ChatRole
  content: string
  image?: string
  status?: "streaming" | "done" | "error"
}

type ChatHistoryItem = {
  role: ChatRole
  content: string
}

const EXAMPLES = [
  "Pick a XIAO board for a battery BLE sensor with low power sleep.",
  "Map the SPI pins for a XIAO W5500 Ethernet Adapter build.",
  "Debug a XIAO board that disappeared from USB after flashing.",
  "Plan a Grove Vision AI V2 trigger workflow for a small robot.",
  "Check the RS485 UART RX/TX and enable pins for a XIAO ESP32C3 expansion board.",
]

const WORKBENCH_TASKS = [
  {
    label: "Choose a board",
    detail: "Compare radios, MCU family, power, and fit.",
    prompt: "Which XIAO should I choose for a battery-powered BLE sensor?",
  },
  {
    label: "Wire an interface",
    detail: "Pin maps for SPI, I2C, UART, Grove, and expansion boards.",
    prompt: "What SPI pins does the XIAO W5500 Ethernet Adapter use?",
  },
  {
    label: "Bring-up debug",
    detail: "USB, bootloader, upload, antenna, and firmware recovery.",
    prompt:
      "My XIAO board is not detected over USB after flashing. What should I check first?",
  },
  {
    label: "Plan a robot",
    detail: "Sense, trigger, LoRa, Jetson, and actuator workflows.",
    prompt:
      "Plan a Seeed robotics demo using Grove Vision AI V2 trigger actions.",
  },
]

const STEPS: Array<{
  key: StepKey
  label: string
  icon: React.ComponentType<{ className?: string }>
}> = [
  { key: "prepare", label: "Prepare", icon: Cpu },
  { key: "retrieve", label: "Retrieve", icon: Database },
  { key: "generate", label: "Generate", icon: Bot },
  { key: "done", label: "Ready", icon: CheckCircle2 },
]

const STEP_RANK = new Map(STEPS.map((step, index) => [step.key, index]))
const INITIAL_ANSWER =
  "Start with a Seeed board, sensor, robotics kit, LoRa module, pinout, firmware recovery issue, or field troubleshooting symptom."

function App() {
  const [question, setQuestion] = React.useState(() => {
    return new URLSearchParams(window.location.search).get("q") ?? ""
  })
  const [messages, setMessages] = React.useState<ChatMessage[]>([])
  const [diagnostics, setDiagnostics] = React.useState<Diagnostics>({})
  const [error, setError] = React.useState("")
  const [isRunning, setIsRunning] = React.useState(false)
  const [image, setImage] = React.useState<string | null>(null)
  const [dragActive, setDragActive] = React.useState(false)
  const abortRef = React.useRef<AbortController | null>(null)
  const fileInputRef = React.useRef<HTMLInputElement | null>(null)
  const chatEndRef = React.useRef<HTMLDivElement | null>(null)
  const questionRef = React.useRef<HTMLTextAreaElement | null>(null)
  const { theme, setTheme } = useTheme()

  const hasUnsavedQuestion = question.trim().length > 0 && !isRunning
  const latestMessageContent =
    messages.length > 0 ? messages[messages.length - 1]?.content : ""

  React.useEffect(() => {
    if (!hasUnsavedQuestion) {
      return undefined
    }

    const handleBeforeUnload = (event: BeforeUnloadEvent) => {
      event.preventDefault()
    }

    window.addEventListener("beforeunload", handleBeforeUnload)
    return () => window.removeEventListener("beforeunload", handleBeforeUnload)
  }, [hasUnsavedQuestion])

  React.useEffect(() => {
    return () => abortRef.current?.abort()
  }, [])

  React.useEffect(() => {
    if (!messages.length) return undefined
    const frame = window.requestAnimationFrame(() => {
      chatEndRef.current?.scrollIntoView({ block: "end" })
    })
    return () => window.cancelAnimationFrame(frame)
  }, [messages.length, latestMessageContent])

  function attachImageFile(file: File | null | undefined) {
    if (!file || !file.type.startsWith("image/")) return
    const reader = new FileReader()
    reader.onload = () => {
      if (typeof reader.result === "string") {
        setImage(reader.result)
        setError("")
      }
    }
    reader.readAsDataURL(file)
  }

  async function handleSubmit(event?: React.FormEvent<HTMLFormElement>) {
    event?.preventDefault()
    if (isRunning) return

    const query = question.trim()
    if (!query && !image) {
      setError("Enter a hardware question or attach a board photo.")
      questionRef.current?.focus()
      return
    }

    abortRef.current?.abort()
    const controller = new AbortController()
    abortRef.current = controller
    const history = chatHistoryForRequest(messages)
    const stagedImage = image
    const userMessage: ChatMessage = {
      id: createMessageId("user"),
      role: "user",
      content: query,
      image: stagedImage ?? undefined,
      status: "done",
    }
    const assistantMessage: ChatMessage = {
      id: createMessageId("assistant"),
      role: "assistant",
      content: "Preparing the request…",
      status: "streaming",
    }
    setIsRunning(true)
    setError("")
    setQuestion("")
    setImage(null)
    setMessages((current) => [...current, userMessage, assistantMessage])
    setDiagnostics({ status: "running", stage: "prepare" })
    updateQueryUrl("")

    try {
      await askBuddy(
        query,
        history,
        stagedImage,
        controller.signal,
        ([nextAnswer, , nextDiagnostics]) => {
          const nextContent = scrubRawSourceIds(nextAnswer || "")
          setMessages((current) =>
            current.map((message) =>
              message.id === assistantMessage.id
                ? {
                    ...message,
                    content: nextContent || message.content,
                    status:
                      nextDiagnostics?.status === "ok" ? "done" : "streaming",
                  }
                : message
            )
          )
          setDiagnostics(nextDiagnostics || {})
        }
      )
    } catch (caught) {
      const message = controller.signal.aborted
        ? "Request stopped."
        : caught instanceof Error
          ? caught.message
          : "The request failed."
      setMessages((current) =>
        current.map((item) =>
          item.id === assistantMessage.id
            ? {
                ...item,
                content:
                  item.content === assistantMessage.content
                    ? message
                    : `${item.content}\n\n${message}`,
                status: "error",
              }
            : item
        )
      )
      if (controller.signal.aborted) {
        setError(message)
      } else {
        setError(message)
      }
    } finally {
      setMessages((current) =>
        current.map((message) =>
          message.id === assistantMessage.id && message.status === "streaming"
            ? { ...message, status: "done" }
            : message
        )
      )
      if (abortRef.current === controller) {
        abortRef.current = null
      }
      setIsRunning(false)
    }
  }

  function stopRequest() {
    abortRef.current?.abort()
  }

  function reset() {
    abortRef.current?.abort()
    setQuestion("")
    setImage(null)
    setMessages([])
    setDiagnostics({})
    setError("")
    updateQueryUrl("")
    questionRef.current?.focus()
  }

  function fillPrompt(prompt: string) {
    setQuestion(prompt)
    setError("")
    questionRef.current?.focus()
  }

  const stage =
    (diagnostics.stage as StepKey | undefined) ??
    (isRunning ? "prepare" : undefined)
  const sourceCount = diagnostics.top_sources?.length ?? 0
  const firstTokenMs =
    diagnostics.agent?.first_token_ms ?? diagnostics.agent_first_token_ms
  const totalMs = diagnostics.timings_ms?.total

  return (
    <main className="workbench-app bg-background text-foreground">
      <a className="skip-link" href="#hardware-question">
        Skip to prompt
      </a>
      <header className="workbench-topbar">
        <div className="workbench-topbar-inner">
          <div className="brand-lockup">
            <span className="brand-mark" aria-hidden="true">
              <Wrench className="size-4" />
            </span>
            <div className="min-w-0">
              <h1>Seeed Project Workbench</h1>
              <p>
                Plan, wire, debug, and verify Seeed electronics projects with
                wiki-backed source peeking.
              </p>
            </div>
          </div>
          <ProgressPanel
            stage={stage}
            diagnostics={diagnostics}
            isRunning={isRunning}
            sourceCount={sourceCount}
            firstTokenMs={firstTokenMs}
            totalMs={totalMs}
          />
          <div className="topbar-actions">
            <Button
              type="button"
              variant="outline"
              size="sm"
              className="topbar-button"
              onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
              aria-label="Toggle color theme"
            >
              <Radio className="size-4" aria-hidden="true" />
              Theme
            </Button>
          </div>
        </div>
      </header>

      <section className="workbench-layout">
        <WorkbenchRail
          messageCount={messages.length}
          sourceCount={sourceCount}
          isRunning={isRunning}
          onPrompt={fillPrompt}
        />

        <section
          className="workbench-main"
          aria-label="Seeed project workbench"
        >
          <div className="workbench-main-header">
            <div className="min-w-0">
              <p className="workbench-kicker">Build log</p>
              <h2>Project conversation</h2>
            </div>
            <span className="session-chip">
              {messages.length
                ? `${Math.ceil(messages.length / 2)} turns`
                : "New build"}
            </span>
          </div>

          <article
            className="chat-panel"
            aria-label="Answer"
            aria-busy={isRunning}
            aria-live="polite"
          >
            <ChatThread
              messages={messages}
              diagnostics={diagnostics}
              endRef={chatEndRef}
            />
          </article>

          <form
            className={cn("chat-composer", dragActive && "composer-dragging")}
            onSubmit={handleSubmit}
            onDragOver={(event) => {
              if (event.dataTransfer?.types?.includes("Files")) {
                event.preventDefault()
                setDragActive(true)
              }
            }}
            onDragLeave={(event) => {
              if (event.currentTarget === event.target) setDragActive(false)
            }}
            onDrop={(event) => {
              event.preventDefault()
              setDragActive(false)
              attachImageFile(event.dataTransfer?.files?.[0])
            }}
          >
            <div className="composer-field">
              <label htmlFor="hardware-question" className="composer-label">
                Workbench prompt
              </label>
              <textarea
                ref={questionRef}
                id="hardware-question"
                name="question"
                value={question}
                onChange={(event) => {
                  setQuestion(event.target.value)
                  if (error) setError("")
                }}
                onKeyDown={(event) => {
                  if (
                    (event.metaKey || event.ctrlKey) &&
                    event.key === "Enter"
                  ) {
                    event.preventDefault()
                    void handleSubmit()
                  }
                }}
                onPaste={(event) => {
                  const file = Array.from(event.clipboardData?.items ?? [])
                    .find((item) => item.type.startsWith("image/"))
                    ?.getAsFile()
                  if (file) {
                    event.preventDefault()
                    attachImageFile(file)
                  }
                }}
                placeholder="Describe the board, sensor, wiring, firmware, symptom, or project goal…"
                rows={3}
                aria-describedby={error ? "question-error" : undefined}
                aria-invalid={Boolean(error)}
                className="composer-input"
              />
              {error ? (
                <p id="question-error" className="composer-error" role="alert">
                  {error}
                </p>
              ) : null}
              {image ? (
                <div className="image-attachment">
                  <img
                    src={image}
                    alt="Attached board photo"
                    className="image-attachment-thumb"
                  />
                  <div className="image-attachment-meta">
                    <span className="image-attachment-title">
                      Board photo attached
                    </span>
                    <span className="image-attachment-hint">
                      Embedded with your question so the assistant can identify
                      the board.
                    </span>
                  </div>
                  <button
                    type="button"
                    className="image-attachment-remove"
                    onClick={() => setImage(null)}
                    aria-label="Remove attached photo"
                  >
                    <X className="size-4" aria-hidden="true" />
                  </button>
                </div>
              ) : null}
            </div>

            <div className="composer-footer">
              <div className="prompt-rail" aria-label="Example prompts">
                {EXAMPLES.map((example) => (
                  <button
                    type="button"
                    key={example}
                    className="prompt-chip"
                    onClick={() => {
                      fillPrompt(example)
                    }}
                  >
                    {example}
                  </button>
                ))}
              </div>

              <div className="composer-actions">
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/*"
                  style={{ display: "none" }}
                  onChange={(event) => {
                    attachImageFile(event.target.files?.[0])
                    event.target.value = ""
                  }}
                />
                <Button
                  type="button"
                  variant="outline"
                  className="composer-attach-button"
                  onClick={() => fileInputRef.current?.click()}
                  aria-label="Attach a board photo"
                >
                  <ImagePlus className="size-4" aria-hidden="true" />
                  Photo
                </Button>
                {messages.length > 0 || question ? (
                  <Button
                    type="button"
                    variant="outline"
                    className="composer-reset-button"
                    onClick={reset}
                    aria-label="Reset chat"
                  >
                    <RotateCcw className="size-4" aria-hidden="true" />
                    Reset
                  </Button>
                ) : null}
                {isRunning ? (
                  <Button
                    type="button"
                    variant="outline"
                    className="composer-stop-button"
                    onClick={stopRequest}
                  >
                    <Square className="size-4" aria-hidden="true" />
                    Stop
                  </Button>
                ) : null}
                <Button
                  type="submit"
                  disabled={isRunning}
                  className="composer-send-button"
                >
                  {isRunning ? (
                    <LoaderCircle
                      className="size-4 animate-spin"
                      aria-hidden="true"
                    />
                  ) : (
                    <Send className="size-4" aria-hidden="true" />
                  )}
                  Ask Workbench
                </Button>
              </div>
            </div>
          </form>
        </section>

        <SourcePeek diagnostics={diagnostics} sourceCount={sourceCount} />
      </section>
    </main>
  )
}

function WorkbenchRail({
  messageCount,
  sourceCount,
  isRunning,
  onPrompt,
}: {
  messageCount: number
  sourceCount: number
  isRunning: boolean
  onPrompt: (prompt: string) => void
}) {
  return (
    <aside className="project-rail" aria-label="Project workbench">
      <section className="rail-section">
        <p className="workbench-kicker">Project brief</p>
        <h2>Build with evidence</h2>
        <p>
          Use this bench to choose Seeed hardware, check pinouts, plan firmware,
          and debug robotics or field troubleshooting issues.
        </p>
      </section>

      <section className="rail-section">
        <p className="rail-section-title">Start from a task</p>
        <div className="task-stack">
          {WORKBENCH_TASKS.map((task) => (
            <button
              key={task.label}
              type="button"
              className="task-card"
              onClick={() => onPrompt(task.prompt)}
            >
              <span>{task.label}</span>
              <small>{task.detail}</small>
            </button>
          ))}
        </div>
      </section>

      <section className="rail-section">
        <p className="rail-section-title">Session</p>
        <dl className="bench-meter">
          <div>
            <dt>Turns</dt>
            <dd>{Math.ceil(messageCount / 2)}</dd>
          </div>
          <div>
            <dt>Sources</dt>
            <dd>{sourceCount}</dd>
          </div>
          <div>
            <dt>Status</dt>
            <dd>{isRunning ? "Running" : "Ready"}</dd>
          </div>
        </dl>
      </section>

      <section className="rail-section checklist-section">
        <p className="rail-section-title">Bench checklist</p>
        <ul>
          <li>Board and exact variant</li>
          <li>Power source and voltage</li>
          <li>Interface pins or bus</li>
          <li>Firmware stack and symptom</li>
        </ul>
      </section>
    </aside>
  )
}

function SourcePeek({
  diagnostics,
  sourceCount,
}: {
  diagnostics: Diagnostics
  sourceCount: number
}) {
  return (
    <aside className="source-peek" aria-label="Source snippets">
      <div className="source-peek-header">
        <div className="source-peek-title">
          <PanelRight className="size-4" aria-hidden="true" />
          <span>Evidence peek</span>
        </div>
        {sourceCount ? (
          <span className="source-count">{sourceCount}</span>
        ) : null}
      </div>
      {sourceCount ? (
        <SourceSnippets diagnostics={diagnostics} />
      ) : (
        <div className="source-empty">
          <BookOpen className="size-4" aria-hidden="true" />
          <p>
            Source snippets appear here as soon as retrieval finds evidence for
            the current project question.
          </p>
        </div>
      )}
    </aside>
  )
}

function SourceSnippets({ diagnostics }: { diagnostics: Diagnostics }) {
  const refs = sourceRefs(diagnostics)
  return (
    <div className="source-snippets">
      <p id="source-snippets" className="source-snippets-title">
        {sourcePanelTitle(diagnostics, refs.length)}
      </p>
      {refs.map((ref) => {
        const safeHref = safeLinkHref(ref.source)
        const title = ref.title || ref.label
        return (
          <section
            key={ref.anchor}
            className="source-snippet"
            id={ref.anchor}
            tabIndex={-1}
            aria-label={`${ref.label} snippet`}
          >
            <p className="source-snippet-kicker">{titleCase(ref.label)}</p>
            <p className="source-snippet-title">
              {safeHref.startsWith("http") ? (
                <a href={safeHref} target="_blank" rel="noreferrer">
                  {title}
                </a>
              ) : (
                title
              )}
            </p>
            {ref.source ? (
              <p className="source-snippet-url">{ref.source}</p>
            ) : null}
            <MarkdownText
              text={ref.snippet}
              className="source-snippet-text source-snippet-markdown"
            />
          </section>
        )
      })}
    </div>
  )
}

function ChatThread({
  messages,
  diagnostics,
  endRef,
}: {
  messages: ChatMessage[]
  diagnostics: Diagnostics
  endRef: React.RefObject<HTMLDivElement | null>
}) {
  if (!messages.length) {
    return (
      <div className="chat-empty">
        <span className="chat-empty-icon" aria-hidden="true">
          <Cpu className="size-5" />
        </span>
        <h2>Start a Seeed build session</h2>
        <p>{INITIAL_ANSWER}</p>
        <div ref={endRef} />
      </div>
    )
  }

  const latestAssistantId = [...messages]
    .reverse()
    .find((message) => message.role === "assistant")?.id

  return (
    <div className="chat-thread">
      {messages.map((message) => {
        const isAssistant = message.role === "assistant"
        const isLatestAssistant = message.id === latestAssistantId
        return (
          <article
            key={message.id}
            className={cn(
              "chat-message",
              isAssistant ? "chat-message-assistant" : "chat-message-user",
              message.status === "error" && "chat-message-error"
            )}
          >
            <div className="chat-message-header">
              <span className="chat-message-role">
                {isAssistant ? "Buddy" : "You"}
              </span>
              {isAssistant && message.status === "streaming" ? (
                <span className="chat-message-status">Streaming…</span>
              ) : null}
              {isAssistant &&
              isLatestAssistant &&
              diagnostics.status === "ok" ? (
                <span className="chat-message-status chat-message-status-ok">
                  Agent answer
                </span>
              ) : null}
            </div>
            {isAssistant ? (
              <MarkdownText text={message.content} />
            ) : (
              <>
                {message.image ? (
                  <img
                    src={message.image}
                    alt="Attached board photo"
                    className="chat-message-image"
                  />
                ) : null}
                {message.content ? (
                  <p className="chat-user-text">{message.content}</p>
                ) : null}
              </>
            )}
          </article>
        )
      })}
      <div ref={endRef} />
    </div>
  )
}

function ProgressPanel({
  stage,
  diagnostics,
  isRunning,
  sourceCount,
  firstTokenMs,
  totalMs,
}: {
  stage?: StepKey
  diagnostics: Diagnostics
  isRunning: boolean
  sourceCount: number
  firstTokenMs?: number
  totalMs?: number
}) {
  const activeRank = stage ? (STEP_RANK.get(stage) ?? -1) : -1
  const budget = evidenceBudget(diagnostics, sourceCount)
  const cacheHit = Boolean(diagnostics.retrieval?.reranker_cache_hit)

  return (
    <section className="progress-panel" aria-label="Run progress">
      <div className="progress-status-row">
        <div className="progress-chips" aria-label="Run summary">
          {budget.label ? (
            <span className={cn("run-chip", `run-chip-${budget.tone}`)}>
              {budget.label}
            </span>
          ) : null}
          {sourceCount ? (
            <span className="run-chip">{budget.sourceLabel}</span>
          ) : null}
          {cacheHit ? <span className="run-chip">Cached rerank</span> : null}
        </div>
        <p className="text-xs text-muted-foreground tabular-nums">
          {isRunning
            ? "Running…"
            : totalMs
              ? `${formatMs(totalMs)} total`
              : "Waiting"}
        </p>
      </div>
      <ol className="progress-list">
        {STEPS.map((step, index) => {
          const state =
            diagnostics.status === "ok" || stage === "done"
              ? "done"
              : index < activeRank
                ? "done"
                : index === activeRank
                  ? "active"
                  : "next"
          const Icon = step.icon
          return (
            <li
              key={step.key}
              className={cn("progress-step", `progress-step-${state}`)}
            >
              <span className="progress-icon">
                <Icon className="size-4" aria-hidden="true" />
              </span>
              <span className="min-w-0">
                <strong>{step.label}</strong>
                <small>
                  {progressDetail(
                    step.key,
                    diagnostics,
                    sourceCount,
                    firstTokenMs,
                    totalMs
                  )}
                </small>
              </span>
            </li>
          )
        })}
      </ol>
    </section>
  )
}

function progressDetail(
  key: StepKey,
  diagnostics: Diagnostics,
  sourceCount: number,
  firstTokenMs?: number,
  totalMs?: number
) {
  if (key === "prepare") return "Input and route"
  if (key === "retrieve") {
    if (sourceCount) return evidenceDetail(diagnostics, sourceCount)
    return "Embedding and HNSW lookup"
  }
  if (key === "generate") {
    const chars = diagnostics.agent?.stream_chars
    if (chars)
      return `${chars} streamed chars${firstTokenMs ? `, first token ${formatMs(firstTokenMs)}` : ""}`
    if (diagnostics.agent?.wait_ms)
      return `Waiting ${formatMs(diagnostics.agent.wait_ms)} for first token`
    if (diagnostics.draft?.chars)
      return `Source draft ${diagnostics.draft.chars} chars`
    return "Agent queued"
  }
  if (key === "done") {
    if (
      (diagnostics.status === "ok" || diagnostics.stage === "done") &&
      totalMs
    ) {
      return `Agent answer in ${formatMs(totalMs)}`
    }
    return "Queued"
  }
  return ""
}

function evidenceBudget(diagnostics: Diagnostics, sourceCount: number) {
  const budget = diagnostics.retrieval?.retrieval_budget
  const mode = budget?.mode ?? ""
  const configuredTop = diagnostics.retrieval?.configured_top_k
  const topK = budget?.top_k ?? diagnostics.retrieval?.top_k
  const sourceLabel =
    configuredTop && topK && configuredTop > topK
      ? `${sourceCount}/${configuredTop} sources`
      : `${sourceCount} sources`

  if (mode === "focused") {
    return {
      label: "Focused evidence",
      sourceLabel,
      tone: "focused",
    }
  }
  if (mode === "full") {
    return {
      label: "Full evidence",
      sourceLabel,
      tone: "full",
    }
  }
  if (mode === "fixed") {
    return {
      label: "Fixed evidence",
      sourceLabel,
      tone: "fixed",
    }
  }
  return {
    label: "",
    sourceLabel,
    tone: "fixed",
  }
}

function evidenceDetail(diagnostics: Diagnostics, sourceCount: number) {
  const backend = diagnostics.retrieval?.vector_index_backend ?? "hnsw"
  const budget = evidenceBudget(diagnostics, sourceCount)
  const parts = [budget.sourceLabel, `via ${backend}`]
  const rerankCount = diagnostics.retrieval?.reranker_candidate_count
  if (rerankCount) parts.push(`${rerankCount} reranked`)
  if (diagnostics.retrieval?.reranker_cache_hit) parts.push("cached")
  return parts.join("; ")
}

function sourcePanelTitle(diagnostics: Diagnostics, count: number) {
  const mode = diagnostics.retrieval?.retrieval_budget?.mode
  if (mode === "focused") return `Focused sources (${count})`
  if (mode === "full") return `Full sources (${count})`
  return `Sources (${count})`
}

function MarkdownText({
  text,
  className,
}: {
  text: string
  className?: string
}) {
  return (
    <div className={cn("answer-markdown", className)}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          a: ({ href = "", children }) => {
            const safeHref = safeLinkHref(href)
            return (
              <a
                href={safeHref}
                target={safeHref.startsWith("http") ? "_blank" : undefined}
                rel={safeHref.startsWith("http") ? "noreferrer" : undefined}
              >
                {children}
              </a>
            )
          },
          table: ({ children }) => (
            <div className="markdown-table-wrap">
              <table>{children}</table>
            </div>
          ),
        }}
      >
        {text}
      </ReactMarkdown>
    </div>
  )
}

type SourceRef = {
  label: string
  anchor: string
  title: string
  source: string
  snippet: string
}

function sourceRefs(diagnostics: Diagnostics): SourceRef[] {
  const sources = diagnostics.top_sources ?? []
  return sources
    .filter(
      (source) => source && (source.title || source.source || source.snippet)
    )
    .map((source, index) => ({
      label: `source ${index + 1}`,
      anchor: `source-snippet-${index + 1}`,
      title: scrubRawSourceIds(source.title?.trim() || `Source ${index + 1}`),
      source: source.source?.trim() || "",
      snippet:
        scrubRawSourceIds(source.snippet?.trim() || "") ||
        "No snippet available.",
    }))
}

function titleCase(value: string) {
  return value.replace(/\b\w/g, (char) => char.toUpperCase())
}

function safeLinkHref(href: string) {
  if (
    href.startsWith("#") ||
    href.startsWith("https://") ||
    href.startsWith("http://")
  ) {
    return href
  }
  return "#"
}

async function askBuddy(
  question: string,
  history: ChatHistoryItem[],
  image: string | null,
  signal: AbortSignal,
  onEvent: (event: AskEvent) => void
) {
  const streamResponse = await fetch(apiUrl("/api/ask"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ image, question, history }),
    signal,
  })
  if (!streamResponse.ok || !streamResponse.body) {
    throw new Error(`Stream failed with ${streamResponse.status}.`)
  }

  const reader = streamResponse.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ""
  let currentEvent = ""

  while (true) {
    const { value, done } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    const lines = buffer.split(/\r?\n/)
    buffer = lines.pop() ?? ""

    for (const rawLine of lines) {
      const line = rawLine.trim()
      if (!line) continue
      if (line.startsWith("event:")) {
        currentEvent = line.slice("event:".length).trim()
        if (currentEvent === "error") {
          throw new Error("The app returned a stream error.")
        }
        continue
      }
      if (!line.startsWith("data:")) continue
      const data = line.slice("data:".length).trim()
      if (currentEvent === "complete" && (!data || data === "null")) {
        return
      }
      const parsed = JSON.parse(data) as unknown
      if (Array.isArray(parsed)) {
        onEvent(parsed as AskEvent)
      }
    }
  }
}

function chatHistoryForRequest(messages: ChatMessage[]): ChatHistoryItem[] {
  return messages
    .filter((message) => {
      return (
        message.status !== "streaming" &&
        message.content.trim().length > 0 &&
        message.content !== INITIAL_ANSWER
      )
    })
    .slice(-8)
    .map((message) => ({
      role: message.role,
      content: message.content.trim().slice(0, 1600),
    }))
}

function createMessageId(prefix: ChatRole) {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
    return `${prefix}-${crypto.randomUUID()}`
  }
  return `${prefix}-${Date.now()}-${Math.random().toString(36).slice(2)}`
}

function apiUrl(path: string) {
  const base = (import.meta.env.VITE_XIAO_BUDDY_API_BASE || "").replace(
    /\/$/,
    ""
  )
  return `${base}${path}`
}

function updateQueryUrl(query: string) {
  const url = new URL(window.location.href)
  if (query) {
    url.searchParams.set("q", query)
  } else {
    url.searchParams.delete("q")
  }
  window.history.replaceState(null, "", url)
}

function scrubRawSourceIds(value: string) {
  return value
    .replace(BRACKETED_RAW_SOURCE_CITATION_RE, (_match, content: string) => {
      const sourceLabel = content.match(/\bsource\s+\d+\b/i)?.[0] ?? "source"
      const sourceNumber = sourceLabel.match(/\d+/)?.[0]
      const anchor = sourceNumber
        ? `#source-snippet-${sourceNumber}`
        : "#source-snippets"
      return `[${sourceLabel.toLowerCase()}](${anchor})`
    })
    .replace(RAW_SOURCE_ID_RE, "source")
}

const RAW_SOURCE_ID_RE =
  /\b(?:wiki-[0-9a-f]{8,}|field-[A-Za-z0-9_.:-]+|xiao-[A-Za-z0-9_.:-]+-(?:identity|pinout|gotchas))\b/gi

const BRACKETED_RAW_SOURCE_CITATION_RE =
  /\[([^\]\n]*(?:wiki-[0-9a-f]{8,}|field-[A-Za-z0-9_.:-]+|xiao-[A-Za-z0-9_.:-]+)[^\]\n]*)\]/gi

function formatMs(value: number) {
  if (value >= 1000) {
    return `${(value / 1000).toFixed(1)}s`
  }
  return `${Math.round(value)}ms`
}

export default App
