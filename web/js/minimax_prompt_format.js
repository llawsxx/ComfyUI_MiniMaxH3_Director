export const PROMPT_IMPORT_HEADER = "# MiniMax Director Prompts v1";
export const PROMPT_IMPORT_MAX_CHARS = 2 * 1024 * 1024;

export class PromptImportError extends Error {
    constructor(key, vars = {}) {
        super(key);
        this.key = key;
        this.vars = vars;
    }
}

export function parsePromptImport(text, { minSeconds, maxSeconds } = {}) {
    if (typeof text !== "string" || text.length > PROMPT_IMPORT_MAX_CHARS) {
        throw new PromptImportError("promptImport.tooLarge");
    }
    const lines = text.replace(/^\uFEFF/, "").replace(/\r\n?/g, "\n").split("\n");
    const records = [];
    let current = null;
    let body = [];
    let hasHeader = false;
    const finish = () => {
        if (!current) return;
        const prompt = body.join("\n").trim();
        if (!prompt) throw new PromptImportError("promptImport.emptyPrompt", { line: current.line });
        records.push({ title: current.title, durationSec: current.durationSec, prompt });
    };
    for (let index = 0; index < lines.length; index++) {
        const line = lines[index];
        const trimmed = line.trim();
        if (!current && !trimmed) continue;
        if (!current && !hasHeader && trimmed === PROMPT_IMPORT_HEADER) {
            hasHeader = true;
            continue;
        }
        if (line.startsWith("## ")) {
            const match = /^##\s+(\d+(?:\.\d)?)s\s*\|\s*(\S.*)$/i.exec(trimmed);
            if (!match) throw new PromptImportError("promptImport.badHeader", { line: index + 1 });
            const durationSec = Number(match[1]);
            if (!Number.isFinite(durationSec) || durationSec <= 0
                || (minSeconds != null && durationSec < minSeconds)
                || (maxSeconds != null && durationSec > maxSeconds)) {
                throw new PromptImportError("promptImport.badDuration", {
                    line: index + 1, min: minSeconds, max: maxSeconds,
                });
            }
            finish();
            current = { title: match[2].trim(), durationSec, line: index + 1 };
            body = [];
        } else if (current) {
            body.push(line.replace(/^\\## /, "## "));
        } else {
            throw new PromptImportError("promptImport.badHeader", { line: index + 1 });
        }
    }
    finish();
    if (!records.length) throw new PromptImportError("promptImport.empty");
    return records;
}
