export function newDirectorProjectId() {
    const bytes = new Uint8Array(16);
    globalThis.crypto.getRandomValues(bytes);
    return Array.from(bytes, (byte) => byte.toString(16).padStart(2, "0")).join("");
}

export function normalizeDirectorProjectId(value) {
    return typeof value === "string" && /^[a-f0-9]{32}$/.test(value) ? value : "";
}

export function directorProjectId(node) {
    const timeline = node?._minimaxEditor?.timeline;
    if (timeline) return normalizeDirectorProjectId(timeline.projectId);
    try {
        const raw = node?.widgets?.find((widget) => widget.name === "timeline_data")?.value;
        const data = JSON.parse(raw || "{}");
        return normalizeDirectorProjectId(data.projectId ?? data.project_id);
    } catch {
        return "";
    }
}

export function setDirectorProjectId(node, projectId = newDirectorProjectId(), timelineData = null) {
    const widget = node?.widgets?.find((item) => item.name === "timeline_data");
    if (!widget) return projectId;
    const saved = node._mmxSavedConfigureData;
    const original = timelineData ?? saved?.widgets_values_named?.timeline_data ?? widget.value;
    let data;
    try {
        data = JSON.parse(original || "{}");
    } catch {
        data = {};
    }
    if (!data || typeof data !== "object" || Array.isArray(data)) data = {};
    const previousProjectId = data.projectId ?? data.project_id;
    data.projectId = projectId;
    delete data.project_id;
    widget.value = JSON.stringify(data);
    if (node._minimaxEditor?.timeline) node._minimaxEditor.timeline.projectId = projectId;
    if (saved) {
        const updated = { ...saved };
        if (saved.widgets_values_named) {
            updated.widgets_values_named = { ...saved.widgets_values_named, timeline_data: widget.value };
        }
        if (Array.isArray(saved.widgets_values)) {
            updated.widgets_values = [...saved.widgets_values];
            let index = saved.widgets_values.indexOf(original);
            if (index < 0 && previousProjectId) {
                index = saved.widgets_values.findIndex((value) => {
                    try {
                        const candidate = JSON.parse(value);
                        return (candidate?.projectId ?? candidate?.project_id) === previousProjectId;
                    } catch {
                        return false;
                    }
                });
            }
            if (index < 0) index = node.widgets.filter((item) => item.serialize !== false).indexOf(widget);
            if (index >= 0) updated.widgets_values[index] = widget.value;
        }
        node._mmxSavedConfigureData = updated;
    }
    return projectId;
}

export function directorEventMatches(node, detail) {
    const projectId = directorProjectId(node);
    return Boolean(projectId && projectId === detail?.project_id && String(node.id) === String(detail?.node_id));
}
