from notion_client import Client
from app.core.config import settings
from app.database import generated_notes_collection

notion = Client(auth=settings.notion_api_key)


def create_user_parent_page(access_token: str, target_tech: str, known_stack: str) -> str:
    client = Client(auth=access_token)
    page_title = f"LearnMate — {target_tech} (from {known_stack})"
    page = client.pages.create(
        parent={"type": "workspace", "workspace": True},
        properties={"title": {"title": [{"text": {"content": page_title}}]}},
    )
    return page["id"]

def split_content(content: str, limit: int = 1900) -> list:
    chunks = []
    while len(content) > limit:
        split_at = content[:limit].rfind("\n")
        if split_at == -1:
            split_at = limit
        chunks.append(content[:split_at])
        content = content[split_at:]
    chunks.append(content)
    return chunks

NOTION_CODE_LANGUAGES = {
    "abap",
    "abc",
    "agda",
    "arduino",
    "assembly",
    "bash",
    "basic",
    "bnf",
    "c",
    "c#",
    "c++",
    "clojure",
    "coffeescript",
    "coq",
    "css",
    "dart",
    "dhall",
    "diff",
    "docker",
    "ebnf",
    "elixir",
    "elm",
    "erlang",
    "f#",
    "flow",
    "fortran",
    "gherkin",
    "glsl",
    "go",
    "graphql",
    "groovy",
    "haskell",
    "hcl",
    "html",
    "idris",
    "java",
    "javascript",
    "json",
    "julia",
    "kotlin",
    "latex",
    "less",
    "lisp",
    "livescript",
    "llvm ir",
    "lua",
    "makefile",
    "markdown",
    "markup",
    "matlab",
    "mathematica",
    "mermaid",
    "nix",
    "notion formula",
    "objective-c",
    "ocaml",
    "pascal",
    "perl",
    "php",
    "plain text",
    "powershell",
    "prolog",
    "protobuf",
    "purescript",
    "python",
    "r",
    "racket",
    "reason",
    "ruby",
    "rust",
    "sass",
    "scala",
    "scheme",
    "scss",
    "shell",
    "smalltalk",
    "solidity",
    "sql",
    "swift",
    "toml",
    "typescript",
    "vb.net",
    "verilog",
    "vhdl",
    "visual basic",
    "webassembly",
    "xml",
    "yaml",
    "java/c/c++/c#",
}

LANGUAGE_ALIASES = {
    "dockerfile": "docker",
    "text": "plain text",
    "plaintext": "plain text",
    "txt": "plain text",
    "md": "markdown",
    "sh": "shell",
    "zsh": "shell",
}

def normalize_code_language(language: str) -> str:
    normalized = (language or "").strip().lower()
    normalized = LANGUAGE_ALIASES.get(normalized, normalized)
    return normalized if normalized in NOTION_CODE_LANGUAGES else "plain text"

def content_to_blocks(content: str) -> list:
    blocks = []
    lines = content.split("\n")
    code_buffer = []
    in_code_block = False
    code_language = "plain text"

    for line in lines:
        if line.startswith("```"):
            if not in_code_block:
                in_code_block = True
                code_language = normalize_code_language(line.replace("```", "").strip() or "plain text")
            else:
                in_code_block = False
                blocks.append({
                    "object": "block",
                    "type": "code",
                    "code": {
                        "language": code_language,
                        "rich_text": [{"type": "text", "text": {"content": "\n".join(code_buffer)[:2000]}}]
                    }
                })
                code_buffer = []
            continue

        if in_code_block:
            code_buffer.append(line)
            continue

        if line.startswith("## "):
            blocks.append({
                "object": "block",
                "type": "heading_2",
                "heading_2": {"rich_text": [{"type": "text", "text": {"content": line[3:].strip()}}]}
            })
        elif line.startswith("# "):
            blocks.append({
                "object": "block",
                "type": "heading_1",
                "heading_1": {"rich_text": [{"type": "text", "text": {"content": line[2:].strip()}}]}
            })
        elif line.startswith("### "):
            blocks.append({
                "object": "block",
                "type": "heading_3",
                "heading_3": {"rich_text": [{"type": "text", "text": {"content": line[4:].strip()}}]}
            })
        elif line.startswith("* ") or line.startswith("- "):
            blocks.append({
                "object": "block",
                "type": "bulleted_list_item",
                "bulleted_list_item": {"rich_text": [{"type": "text", "text": {"content": line[2:].strip()}}]}
            })
        elif line.strip() == "":
            continue
        else:
            chunks = split_content(line)
            for chunk in chunks:
                blocks.append({
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {"rich_text": [{"type": "text", "text": {"content": chunk}}]}
                })

    return blocks

async def create_notion_topic(
    title: str,
    content: str,
    session_id: str = None,
    notion_token: str = None,
    page_id: str = None,
):
    auth_token = notion_token if notion_token else settings.notion_api_key
    notion_client = Client(auth=auth_token)

    parent_id = (
        page_id
        if page_id
        else settings.notion_parent_page_id
    )

    blocks = content_to_blocks(content)

    new_page = {
        "parent": {"page_id": parent_id},
        "properties": {
            "title": {
                "title": [{"text": {"content": title}}]
            }
        },
        "children": blocks,
    }

    response = notion_client.pages.create(**new_page)
    url = response["url"]

    if session_id:
        await generated_notes_collection.insert_one({
            "session_id": session_id,
            "title": title,
            "content": content,
            "url": url,
        })

    return url
