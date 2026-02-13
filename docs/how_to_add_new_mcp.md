# How To Add Any MCP Server (Codex + VS Code Remote SSH)

This document is a reusable recipe for adding and debugging any MCP server.
It is based on the workflow that worked for `wandb`, but all steps are generic.

## 1) Add the MCP server in Codex config

Edit `/home/cizinsky/.codex/config.toml` and add a block like:

```toml
[mcp_servers.<SERVER_NAME>]
url = "https://<MCP_HOST>/mcp"
bearer_token_env_var = "<TOKEN_ENV_VAR>"
enabled = true
```

Notes:
- `<SERVER_NAME>` is the name you will use when testing (for example: `wandb`, `notion`).
- `bearer_token_env_var` is the environment variable name that must exist in the process where Codex runs.

## 2) Set the token in the right process environment

This is the most common failure point.

If Codex runs in VS Code Remote SSH, the extension host usually does **not** inherit `~/.bashrc` from your interactive shell.

Preferred setup for VS Code Remote SSH:
- Create/update `~/.vscode-server/server-env-setup`
- Add:

```bash
export <TOKEN_ENV_VAR>="<YOUR_TOKEN>"
```

For example (wandb):

```bash
export WANDB_MCP_TOKEN="<YOUR_TOKEN>"
```

Security:
- Never paste real tokens into chat/logs.
- If a token is exposed, rotate it immediately.

## 3) Restart VS Code Remote server/session

After changing `server-env-setup`, restart the remote VS Code environment:

1. In VS Code Command Palette: `Developer: Reload Window`
2. If that does not work, fully restart the remote VS Code server and reconnect.

Important (observed in practice):
- Adding a **new** MCP server or a **new env var** often requires a full remote server restart.
- A window reload alone may leave the existing extension host alive with stale env vars.
- If `... env var is not set` persists after reload, restart the remote server process and reconnect.

## 4) Verify MCP startup from Codex

Test whether Codex can start the server:
- `list_mcp_resources(server="<SERVER_NAME>")`

Interpretation:
- If you get `Environment variable <TOKEN_ENV_VAR> ... is not set`, environment propagation is still broken.
- If it returns successfully (even `resources: []`), the MCP server is up.

Important:
- `resources: []` can be valid. Some MCP servers expose tools but not static resources.

## 5) Verify functional access (not just startup)

After startup succeeds, run one real server-specific call.

Examples:
- For Notion: search/fetch a page.
- For W&B: query projects for your entity.

This confirms auth + permissions + tool path, not only process startup.

## 6) Fast troubleshooting checklist

If MCP still fails:
- Confirm the config block exists and is enabled in `/home/cizinsky/.codex/config.toml`.
- Confirm the environment variable name in config matches exactly the exported name.
- Confirm `~/.vscode-server/server-env-setup` exists on the **remote** host (not local machine).
- Confirm token is non-empty in the remote extension host environment.
- Reload/restart remote VS Code after any env change.
- If needed, force a full server restart from remote shell:

```bash
pkill -f '/home/$USER/.vscode-server/cli/servers/Stable-.*/server/out/server-main.js'
```

Then reconnect from VS Code Remote SSH.
- Retry `list_mcp_resources(server="<SERVER_NAME>")`.

## 7) Minimal end-to-end template

1. Add `[mcp_servers.<SERVER_NAME>]` in Codex config.
2. Set `export <TOKEN_ENV_VAR>=...` in `~/.vscode-server/server-env-setup`.
3. Reload/restart VS Code Remote SSH session.
4. Run `list_mcp_resources(server="<SERVER_NAME>")`.
5. Run one real MCP tool call for that server.
