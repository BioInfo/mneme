# Claude Code Session JSONL Format Reference

## Overview

Claude Code stores session history in JSONL (JSON Lines) files at:
- **Mac:** `~/.claude/projects/{project-path}/*.jsonl`
- **Session files:** `{uuid}.jsonl` (main sessions)
- **Agent files:** `agent-{short-id}.jsonl` (sub-agent sessions)

---

## File Naming

| Pattern | Description |
|---------|-------------|
| `{uuid}.jsonl` | Main session file (e.g., `e44fe688-94cb-435f-9162-361a0097e089.jsonl`) |
| `agent-{7char}.jsonl` | Sub-agent session (e.g., `agent-a0f9f8d.jsonl`) |

---

## Record Types

Each line in the JSONL file is a JSON object with a `type` field:

### 1. Summary (`type: "summary"`)

Session summary, typically at the start of the file.

```json
{
  "type": "summary",
  "summary": "Email organization & nightly automation setup complete",
  "leafUuid": "d5540c4d-551b-49c1-b521-fe187f93d04d"
}
```

**Fields:**
- `summary`: Brief description of session work (HIGH VALUE for indexing)
- `leafUuid`: UUID of the leaf message this summarizes

**Indexing:** Always index. This is the most valuable content.

---

### 2. User Message (`type: "user"`)

User input to Claude.

```json
{
  "type": "user",
  "parentUuid": "3267b823-05e1-4953-809d-f8b57794439d",
  "uuid": "fcc68f65-34a6-47a1-b7f5-93f05acb1091",
  "sessionId": "e44fe688-94cb-435f-9162-361a0097e089",
  "timestamp": "2025-12-19T12:07:37.862Z",
  "cwd": "/Users/bioinfo",
  "message": {
    "role": "user",
    "content": "did the reindexing and everything run well overnight?"
  }
}
```

**Fields:**
- `message.content`: Can be string OR array (for tool results)
- `sessionId`: Links to session
- `timestamp`: When sent
- `uuid`: Unique message ID

**Content Variants:**

Simple text:
```json
"content": "What did we do yesterday?"
```

With tool results:
```json
"content": [
  {
    "type": "tool_result",
    "tool_use_id": "toolu_0113qnd9u7LRfj23HJ8Wh7sN",
    "content": "command output here...",
    "is_error": false
  }
]
```

**Indexing:** Index text content. Skip tool results (too noisy).

---

### 3. Assistant Message (`type: "assistant"`)

Claude's response, including text, thinking, and tool calls.

```json
{
  "type": "assistant",
  "parentUuid": "fcc68f65-34a6-47a1-b7f5-93f05acb1091",
  "uuid": "0064a4ed-4f94-403c-b241-e63b719d7fe5",
  "sessionId": "e44fe688-94cb-435f-9162-361a0097e089",
  "timestamp": "2025-12-19T12:07:46.116Z",
  "message": {
    "model": "claude-opus-4-5-20251101",
    "role": "assistant",
    "content": [
      {"type": "text", "text": "Let me check the overnight logs."},
      {"type": "thinking", "thinking": "I should check the LaunchAgent logs..."},
      {"type": "tool_use", "id": "toolu_xxx", "name": "Bash", "input": {...}}
    ]
  }
}
```

**Content Block Types:**

| Type | Description | Index? |
|------|-------------|--------|
| `text` | Claude's actual response | Yes |
| `thinking` | Internal reasoning (extended thinking) | No (verbose) |
| `tool_use` | Tool invocation request | No |

**Indexing:** Index only `type: "text"` blocks.

---

### 4. File History Snapshot (`type: "file-history-snapshot"`)

Tracks file changes during session.

```json
{
  "type": "file-history-snapshot",
  "messageId": "fcc68f65-34a6-47a1-b7f5-93f05acb1091",
  "snapshot": {
    "messageId": "fcc68f65-34a6-47a1-b7f5-93f05acb1091",
    "trackedFileBackups": {},
    "timestamp": "2025-12-19T12:07:37.871Z"
  },
  "isSnapshotUpdate": false
}
```

**Indexing:** Skip. No semantic value.

---

## Common Fields

| Field | Type | Description |
|-------|------|-------------|
| `uuid` | string | Unique message identifier |
| `parentUuid` | string | Parent message (conversation threading) |
| `sessionId` | string | Session UUID |
| `timestamp` | ISO 8601 | When the message was created |
| `cwd` | string | Working directory at message time |
| `version` | string | Claude Code version (e.g., "2.0.72") |
| `gitBranch` | string | Current git branch (often empty) |
| `isSidechain` | boolean | Whether this is a sidechain message |

---

## Extraction Strategy

### Priority 1: Summaries
```python
if record['type'] == 'summary':
    yield record['summary']
```

### Priority 2: User Messages (text only)
```python
if record['type'] == 'user':
    content = record['message']['content']
    if isinstance(content, str):
        yield content
    # Skip tool results in content arrays
```

### Priority 3: Assistant Text
```python
if record['type'] == 'assistant':
    for block in record['message']['content']:
        if block['type'] == 'text':
            yield block['text']
```

---

## Sample File Statistics

Based on analysis of `~/.claude/projects/-Users-bioinfo/`:

| Metric | Value |
|--------|-------|
| Total session files | ~400 |
| Avg file size | 200 KB |
| Largest file | 4 MB |
| Messages per session | 20-100 |
| Summaries per file | 1-3 |

---

## Edge Cases

1. **Empty sessions:** Files with only `file-history-snapshot` records
2. **Tool-heavy sessions:** Mostly tool calls, few text responses
3. **Interrupted sessions:** May lack summary
4. **Agent sessions:** Typically smaller, focused sub-tasks
5. **Very long responses:** Assistant text > 10KB should be chunked

---

## Example: Full Message Flow

```
1. file-history-snapshot (session init)
2. user message: "did the reindexing run well?"
3. assistant thinking: "I should check logs..."
4. assistant text: "Let me check the logs."
5. assistant tool_use: Bash command
6. user message (tool_result): command output
7. assistant thinking: "The output shows..."
8. assistant text: "The indexing completed successfully."
9. summary: "Vector DB indexing verification"
```

**What to index from this flow:**
- Line 2: User question (captures intent)
- Lines 4, 8: Assistant text (captures outcomes)
- Line 9: Summary (captures essence)
