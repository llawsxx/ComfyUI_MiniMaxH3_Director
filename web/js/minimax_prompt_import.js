import { t } from "./minimax_i18n.js";
import { isVideoBatchTask, minDurationSec, maxDurationSec, newBatchSegment } from "./minimax_gen_timeline.js";
import { flushBatchPromptInputs } from "./minimax_image_batch.js";
import { flushFl2vPromptDraft, newFl2vShot, updateFl2vDetailUI } from "./minimax_fl2v.js";
import { parsePromptImport, PROMPT_IMPORT_HEADER, PROMPT_IMPORT_MAX_CHARS, PromptImportError } from "./minimax_prompt_format.js";

export const PROMPT_IMPORT_STYLES = `
.bd-prompt-import{max-width:660px;min-height:0;overflow:auto}
.bd-prompt-import textarea{width:100%;min-height:180px;flex:1 1 240px;box-sizing:border-box;resize:vertical;background:#141414;color:#eee;border:1px solid #444;border-radius:4px;padding:8px;font:12px/1.5 monospace}
.bd-prompt-import-options{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
.bd-prompt-import-summary{color:#9ab;white-space:pre-wrap}
.bd-prompt-import-summary.error{color:#f88}
`;

function supportsPromptImport(editor) {
    return !!(editor.isFl2vMode?.() || (editor.isImageBatch?.() && isVideoBatchTask(editor.getTaskKey?.())));
}

function hasExternalGroups(editor) {
    return !!(editor.hasExternalI2vGroups?.() || editor.hasExternalR2vGroups?.());
}

export function syncPromptImportButton(editor) {
    const button = editor.root?.querySelector('[data-a="prompt-import"]');
    if (!button) return;
    button.classList.toggle("hidden", !supportsPromptImport(editor));
    button.disabled = hasExternalGroups(editor);
}

export function applyPromptImport(editor, records, mode) {
    if (!supportsPromptImport(editor) || hasExternalGroups(editor)) {
        throw new PromptImportError("promptImport.unsupported");
    }
    if (!["update", "append", "replace"].includes(mode)) throw new Error("Invalid import mode");
    editor._flushPromptTokenEditors?.();
    flushBatchPromptInputs(editor);
    flushFl2vPromptDraft(editor);
    for (const input of editor.batchList?.querySelectorAll("input[data-batch-sec-index]") || []) {
        clearTimeout(input._t);
        input._t = null;
    }
    const isFl2v = editor.isFl2vMode();
    const source = isFl2v ? editor.timeline.shots || [] : editor.timeline.segments || [];
    const offset = mode === "append" ? source.length : 0;
    const next = mode === "replace" ? [] : source.slice();
    records.forEach((record, index) => {
        const target = offset + index;
        const existing = mode === "update" ? source[target] : null;
        const fields = { ...existing, ...record, previewB64: "", previewFrames: [] };
        next[target] = isFl2v
            ? newFl2vShot(fields)
            : newBatchSegment({ ...fields, frameRate: editor.getFrameRate(), taskType: fields.taskType || editor.getTaskKey() });
    });
    const suspended = editor._suspendPromptFlush;
    editor._suspendPromptFlush = true;
    try {
        if (isFl2v) {
            editor.timeline.shots = next;
            editor._fl2vPromptSegIndex = null;
            editor._fl2vPromptShotId = null;
        } else {
            editor.timeline.segments = next;
        }
        editor._previewSegments = null;
        editor.selectedIndex = offset;
        if (mode === "replace") {
            editor.timeline.runSelection = next.map((record, index) => index);
        } else if (editor.timeline.runSelectEnabled && next.length > source.length) {
            editor.timeline.runSelection = [
                ...(editor.timeline.runSelection || []),
                ...next.slice(source.length).map((record, index) => source.length + index),
            ];
        }
        editor.commit(false, { syncTimeline: true });
        if (isFl2v) updateFl2vDetailUI(editor);
        editor.updateVideoNameLabel?.();
        editor.updateDomWidgetHeight?.();
        editor._writeTimelineWidget?.();
    } finally {
        editor._suspendPromptFlush = suspended;
    }
}

export function openPromptImport(editor) {
    if (!supportsPromptImport(editor) || hasExternalGroups(editor)) return;
    editor._closeBdModal();
    const overlay = document.createElement("div");
    overlay.className = "bd-modal-overlay";
    const panel = document.createElement("div");
    panel.className = "bd-modal bd-prompt-import";
    const title = document.createElement("div");
    title.className = "bd-modal-title";
    title.textContent = t("toolbar.importPrompts");
    const hint = document.createElement("div");
    hint.className = "bd-modal-body";
    hint.textContent = t("promptImport.hint");
    const options = document.createElement("div");
    options.className = "bd-prompt-import-options";
    const chooseFile = document.createElement("button");
    chooseFile.type = "button";
    chooseFile.className = "bd-btn";
    chooseFile.textContent = t("promptImport.chooseFile");
    const fileInput = document.createElement("input");
    fileInput.type = "file";
    fileInput.accept = ".txt,.md,text/plain,text/markdown";
    fileInput.hidden = true;
    const mode = document.createElement("select");
    mode.className = "bd-select";
    for (const value of ["update", "append", "replace"]) {
        const option = document.createElement("option");
        option.value = value;
        option.textContent = t(`promptImport.${value}`);
        mode.appendChild(option);
    }
    options.append(chooseFile, fileInput, mode);
    const input = document.createElement("textarea");
    input.placeholder = `${PROMPT_IMPORT_HEADER}\n\n## 5s | ${t("promptImport.exampleTitle")}\n${t("promptImport.examplePrompt")}`;
    input.setAttribute("aria-label", t("toolbar.importPrompts"));
    const summary = document.createElement("div");
    summary.className = "bd-prompt-import-summary";
    summary.setAttribute("role", "status");
    const actions = document.createElement("div");
    actions.className = "bd-modal-actions";
    const cancel = document.createElement("button");
    cancel.type = "button";
    cancel.className = "bd-btn";
    cancel.textContent = t("dialog.cancel");
    const confirm = document.createElement("button");
    confirm.type = "button";
    confirm.className = "bd-btn bd-btn-primary";
    const showError = (error) => {
        summary.classList.add("error");
        summary.textContent = error instanceof PromptImportError ? t(error.key, error.vars) : String(error.message || error);
        confirm.disabled = true;
    };
    const validate = () => {
        confirm.textContent = t(mode.value === "replace" ? "promptImport.confirmReplace" : "promptImport.confirm");
        try {
            const fps = editor.getFrameRate();
            const records = parsePromptImport(input.value, { minSeconds: minDurationSec(fps), maxSeconds: maxDurationSec(fps) });
            summary.classList.remove("error");
            summary.textContent = t("promptImport.summary", {
                n: records.length,
                seconds: Math.round(records.reduce((total, record) => total + record.durationSec, 0) * 10) / 10,
            }) + "\n" + t(`promptImport.${mode.value}Hint`);
            confirm.disabled = false;
            return records;
        } catch (error) {
            showError(error);
            return null;
        }
    };
    input.addEventListener("input", validate);
    mode.addEventListener("change", validate);
    chooseFile.onclick = () => fileInput.click();
    fileInput.onchange = async () => {
        const file = fileInput.files?.[0];
        if (!file) return;
        try {
            if (file.size > PROMPT_IMPORT_MAX_CHARS) throw new PromptImportError("promptImport.tooLarge");
            input.value = await file.text();
            validate();
        } catch (error) {
            showError(error);
        } finally {
            fileInput.value = "";
        }
    };
    cancel.onclick = () => editor._closeBdModal();
    confirm.onclick = () => {
        const records = validate();
        if (!records) return;
        try {
            applyPromptImport(editor, records, mode.value);
            editor._closeBdModal();
        } catch (error) {
            showError(error);
        }
    };
    overlay.onclick = (event) => {
        if (event.target === overlay) editor._closeBdModal();
    };
    panel.addEventListener("wheel", (event) => event.stopPropagation(), { passive: true });
    panel.addEventListener("pointerdown", (event) => event.stopPropagation());
    editor._modalKeyHandler = (event) => {
        if (event.key === "Escape") {
            event.preventDefault();
            event.stopPropagation();
            editor._closeBdModal();
        } else if (panel.contains(event.target)) {
            event.stopPropagation();
        }
    };
    window.addEventListener("keydown", editor._modalKeyHandler, true);
    actions.append(cancel, confirm);
    panel.append(title, hint, options, input, summary, actions);
    overlay.appendChild(panel);
    editor.root.appendChild(overlay);
    editor._modalEl = overlay;
    validate();
    input.focus();
}

export function bindPromptImport(editor) {
    const button = editor.root?.querySelector('[data-a="prompt-import"]');
    if (button) button.onclick = () => openPromptImport(editor);
    syncPromptImportButton(editor);
}
