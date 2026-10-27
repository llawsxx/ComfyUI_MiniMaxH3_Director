import { api } from "../../scripts/api.js";
import { t } from "./minimax_i18n.js";

export const GUIDE_FRAME_STYLES = `
.bd-guide-frames{display:flex;flex-direction:column;gap:6px;min-width:0;padding:8px;border:1px solid #353535;border-radius:8px;background:#121212}
.bd-guide-head{display:flex;align-items:center;justify-content:space-between;gap:8px;font-size:12px;color:#ddd}
.bd-guide-hint{font-size:11px;color:#999;line-height:1.4;white-space:pre-line}
.bd-guide-list{display:flex;flex-direction:column;gap:6px;max-height:260px;overflow:auto;overscroll-behavior:contain}
.bd-guide-row{display:flex;align-items:center;flex-wrap:wrap;gap:6px;padding:4px;background:#1b1b1b;border-radius:6px}
.bd-guide-image{display:flex;align-items:center;justify-content:center;flex:0 0 160px;max-width:100%;height:100px;box-sizing:border-box;padding:0;overflow:hidden;border:1px dashed #666;border-radius:4px;background:#242424;color:#bbb;cursor:pointer;font-size:11px}
.bd-guide-image img{width:100%;height:100%;object-fit:contain}
.bd-guide-row select,.bd-guide-row input{box-sizing:border-box;padding:4px;border:1px solid #444;border-radius:4px;background:#191919;color:#eee;font-size:12px}
.bd-guide-row input{width:76px}
.bd-guide-row input:invalid{border-color:#e66}
.bd-guide-row .bd-guide-frame-label{font-size:11px;color:#aaa}
.bd-guide-row .bd-guide-remove{margin-left:auto}
`;

export function cloneGuideFrames(entries) {
    return Array.isArray(entries) ? entries.map((entry) => ({ ...entry })) : [];
}

function guideImageUrl(imageFile) {
    const path = String(imageFile).replace(/\\/g, "/");
    const split = path.lastIndexOf("/");
    const params = new URLSearchParams({ filename: path.slice(split + 1), type: "input" });
    if (split >= 0) params.set("subfolder", path.slice(0, split));
    return api.apiURL(`/view?${params}`);
}

export function createGuideFramesEditor(editor, {
    getTarget,
    getFrameCount = () => 0,
    positions = ["first", "middle", "last", "frame"],
    common = false,
    onChange = () => editor.commit?.(true, { syncTimeline: true }),
} = {}) {
    const root = document.createElement("div");
    root.className = "bd-guide-frames";
    const head = document.createElement("div");
    head.className = "bd-guide-head";
    const title = document.createElement("b");
    title.textContent = t(common ? "guides.commonTitle" : "guides.title");
    const add = document.createElement("button");
    add.type = "button";
    add.className = "bd-btn bd-guide-add";
    add.textContent = t("guides.add");
    head.append(title, add);
    const hint = document.createElement("div");
    hint.className = "bd-guide-hint";
    hint.textContent = t(common ? "guides.commonHint" : "guides.hint");
    const list = document.createElement("div");
    list.className = "bd-guide-list";
    root.append(head, hint, list);
    root.addEventListener("click", (event) => event.stopPropagation());
    root.addEventListener("keydown", (event) => event.stopPropagation());
    root.addEventListener("wheel", (event) => event.stopPropagation(), { passive: true });
    const entries = () => getTarget()?.guideFrames || [];
    const liveGuide = (id) => entries().find((entry) => entry.id === id);
    const frameRate = () => {
        const value = Number(editor.getFrameRate?.() ?? editor.timeline?.frameRate ?? 24);
        return Number.isFinite(value) && value > 0 ? value : 24;
    };
    const secondsForFrame = (frameIndex) => String(Number((Number(frameIndex || 0) / frameRate()).toFixed(6)));
    const changed = () => {
        onChange();
        editor.updateDomWidgetHeight?.();
    };
    const render = () => {
        list.replaceChildren();
        for (const guide of entries()) {
            if (!guide.id) guide.id = crypto.randomUUID();
            const id = guide.id;
            const row = document.createElement("div");
            row.className = "bd-guide-row";
            row.dataset.guideId = id;
            const image = document.createElement("button");
            image.type = "button";
            image.className = "bd-guide-image";
            image.title = t("guides.pick");
            if (guide.imageFile || guide.imageB64) {
                const thumbnail = document.createElement("img");
                thumbnail.alt = t("guides.pick");
                thumbnail.src = guide.imageFile
                    ? guideImageUrl(guide.imageFile)
                    : (guide.imageB64.startsWith("data:") ? guide.imageB64 : `data:image/png;base64,${guide.imageB64}`);
                image.appendChild(thumbnail);
            } else {
                image.textContent = t("guides.pick");
            }
            image.onclick = async () => {
                image.disabled = true;
                try {
                    const picked = await editor.chooseImageInput({ title: t("guides.pick"), currentValue: liveGuide(id)?.imageFile || "" });
                    const current = liveGuide(id);
                    if (!picked?.imageFile || !current) return;
                    current.imageFile = picked.imageFile;
                    current.imageB64 = "";
                    changed();
                    render();
                } catch (error) {
                    alert(t("upload.alertFailed", { err: error.message || error }));
                } finally {
                    image.disabled = false;
                }
            };
            const position = document.createElement("select");
            position.className = "bd-guide-position";
            position.setAttribute("aria-label", t("guides.position"));
            for (const value of positions) {
                const option = document.createElement("option");
                option.value = value;
                option.textContent = t(`guides.${value}`);
                position.appendChild(option);
            }
            position.value = positions.includes(guide.position) ? guide.position : positions[0];
            const frame = document.createElement("input");
            frame.className = "bd-guide-frame-index";
            frame.type = "number";
            frame.min = "0";
            frame.step = "1";
            frame.value = String(guide.frameIndex ?? 0);
            frame.setAttribute("aria-label", t("guides.frameIndex"));
            const total = Number(getFrameCount()) || 0;
            if (!common && total > 0) frame.title = t("guides.rangeHint", { n: total - 1 });
            frame.hidden = position.value !== "frame";
            const frameLabel = document.createElement("span");
            frameLabel.className = "bd-guide-frame-label";
            frameLabel.textContent = t("guides.frameIndex");
            frameLabel.hidden = frame.hidden;
            const seconds = document.createElement("input");
            seconds.className = "bd-guide-seconds";
            seconds.type = "number";
            seconds.min = "0";
            seconds.step = "any";
            seconds.value = secondsForFrame(guide.frameIndex);
            seconds.setAttribute("aria-label", t("guides.seconds"));
            seconds.title = t("guides.secondsHint", { fps: frameRate() });
            seconds.hidden = frame.hidden;
            const secondsLabel = document.createElement("span");
            secondsLabel.className = "bd-guide-frame-label";
            secondsLabel.textContent = t("guides.seconds");
            secondsLabel.hidden = frame.hidden;
            position.onchange = () => {
                const current = liveGuide(id);
                if (!current) return;
                current.position = position.value;
                frame.hidden = position.value !== "frame";
                frameLabel.hidden = frame.hidden;
                seconds.hidden = frame.hidden;
                secondsLabel.hidden = frame.hidden;
                changed();
            };
            frame.oninput = () => {
                const current = liveGuide(id);
                if (!current || !frame.checkValidity() || frame.value === "") return;
                const frameIndex = Number(frame.value);
                if (!Number.isSafeInteger(frameIndex)) return;
                current.frameIndex = frameIndex;
                seconds.value = secondsForFrame(frameIndex);
                editor.scheduleTimelineSync?.();
            };
            frame.onchange = () => {
                if (frame.reportValidity() && frame.value !== "" && Number.isSafeInteger(Number(frame.value))) changed();
            };
            seconds.oninput = () => {
                const current = liveGuide(id);
                if (!current || !seconds.checkValidity() || seconds.value === "") return;
                const frameIndex = Math.round(Number(seconds.value) * frameRate());
                if (!Number.isSafeInteger(frameIndex)) return;
                current.frameIndex = frameIndex;
                frame.value = String(frameIndex);
                editor.scheduleTimelineSync?.();
            };
            seconds.onchange = () => {
                if (!seconds.reportValidity() || seconds.value === "" || !liveGuide(id)) return;
                const frameIndex = Math.round(Number(seconds.value) * frameRate());
                if (!Number.isSafeInteger(frameIndex)) return;
                seconds.value = secondsForFrame(liveGuide(id).frameIndex);
                changed();
            };
            const remove = document.createElement("button");
            remove.type = "button";
            remove.className = "bd-btn bd-guide-remove";
            remove.textContent = t("guides.remove");
            remove.onclick = () => {
                const target = getTarget();
                if (!target) return;
                target.guideFrames = entries().filter((entry) => entry.id !== id);
                changed();
                render();
            };
            row.append(image, position, frameLabel, frame, secondsLabel, seconds, remove);
            list.appendChild(row);
        }
    };
    add.onclick = () => {
        const target = getTarget();
        if (!target) return;
        target.guideFrames = [...entries(), {
            id: crypto.randomUUID(), position: positions[0], frameIndex: 0, imageFile: "",
        }];
        changed();
        render();
    };
    render();
    return root;
}
