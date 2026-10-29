/* Render only the explicit public presentation schema; evidence is always text.
 * Native DOM methods avoid interpreting model output as HTML.
 * https://developer.mozilla.org/en-US/docs/Web/API/Node/textContent
 * https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API/Using_Fetch
 */
"use strict";

const byId = (id) => document.getElementById(id);
const statusLabels = { accepted: "Accepted", failed: "Failed acceptance", interrupted: "Interrupted", deferred: "Deferred · not run" };
const metricLabels = { recall: "Recall", specificity: "Near-name specificity", retention: "Knowledge retention", nonempty: "Non-empty" };
const categoryLabels = { fact_recall: "Fact recall", near_name_safety: "Near-name specificity", near_name_negative: "Near-name specificity", near_name: "Near-name specificity", common_knowledge: "Knowledge retention", common_knowledge_controls: "Knowledge retention" };
let dataset;
let selectedRun;

function element(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
}

function option(value, label) {
  const node = element("option", label);
  node.value = value;
  return node;
}

function fraction(metric) {
  return metric ? `${metric.passed}/${metric.total}` : "—";
}

function badge(status) {
  return element("span", statusLabels[status] || status, `badge ${status}`);
}

function sourceList(target, sources) {
  target.replaceChildren();
  for (const source of sources) {
    const item = element("li");
    let url;
    try { url = new URL(source.url); } catch { url = null; }
    if (url && url.protocol === "https:") {
      const link = element("a", source.path);
      link.href = url.href;
      item.append(link);
    } else {
      item.append(element("span", source.path));
    }
    item.append(element("code", `SHA-256 ${source.sha256}`, "hash"));
    target.append(item);
  }
}

function studyRuns() {
  const study = byId("study-filter").value;
  return dataset.runs.filter((run) => study === "all" || run.study_id === study);
}

function visibleRuns() {
  const recipe = byId("recipe-filter").value;
  return studyRuns().filter((run) => recipe === "all" || run.id === recipe);
}

function metricRow(label, metrics, status, run) {
  const row = element("tr", undefined, run ? "" : "baseline-row");
  const heading = element("th");
  heading.scope = "row";
  if (run) {
    const button = element("button", label, "inspect");
    button.type = "button";
    button.dataset.runId = run.id;
    button.setAttribute("aria-label", `Inspect ${label}`);
    button.setAttribute("aria-pressed", String(selectedRun?.id === run.id));
    button.addEventListener("click", () => {
      selectRun(run);
      byId("run-title").tabIndex = -1;
      byId("run-title").focus();
    });
    heading.append(button);
  } else {
    heading.textContent = label;
  }
  row.append(heading);
  const state = element("td");
  state.append(run ? badge(status) : element("span", "Untouched base", "quiet"));
  row.append(state);
  for (const key of Object.keys(metricLabels)) row.append(element("td", fraction(metrics?.[key]), "metric"));
  return row;
}

function renderTables() {
  const runs = visibleRuns();
  const target = byId("study-tables");
  target.replaceChildren();
  for (const study of dataset.studies) {
    const group = runs.filter((run) => run.study_id === study.id);
    if (!group.length) continue;
    const section = element("section", undefined, "study");
    const title = element("h3", study.label);
    title.id = `study-${study.id}`;
    section.setAttribute("aria-labelledby", title.id);
    section.append(title, element("p", `Pinned base: ${study.model_id}`, "quiet"), element("p", `Revision: ${study.model_revision}`, "identity"));
    const scroller = element("div", undefined, "table-scroll");
    scroller.tabIndex = 0;
    scroller.setAttribute("aria-label", `${study.label} score comparison; scroll horizontally on small screens`);
    const table = element("table");
    const head = element("thead");
    const headers = element("tr");
    for (const label of ["Recipe · select to inspect", "Outcome", ...Object.values(metricLabels)]) {
      const cell = element("th", label);
      cell.scope = "col";
      headers.append(cell);
    }
    head.append(headers);
    const body = element("tbody");
    const baseline = dataset.runs.find((run) => run.study_id === study.id && run.baseline)?.baseline;
    if (baseline) body.append(metricRow("Baseline", baseline));
    for (const run of group) body.append(metricRow(run.label, run.tuned, run.status, run));
    table.append(head, body);
    scroller.append(table);
    section.append(scroller);
    if (group.some((run) => !run.tuned)) section.append(element("p", "— means no final tuned evaluation. Interrupted and deferred recipes retain their distinct states.", "quiet"));
    target.append(section);
  }
  byId("filter-status").textContent = `Showing ${runs.length} of ${dataset.runs.length} registered recipes. Select a recipe name to inspect its records.`;
}

function renderLoss(run) {
  const hasLoss = run.history.some((point) => point.loss !== undefined || point.eval_loss !== undefined);
  byId("loss-figure").hidden = !hasLoss;
  byId("loss-values").hidden = !hasLoss;
  byId("loss-empty").hidden = hasLoss;
  byId("loss-values").open = false;
  byId("loss-rows").replaceChildren();
  if (!hasLoss) {
    byId("loss-empty").textContent = run.status === "deferred" ? "This recipe was not run; there is no loss history." : "No loss history is present in the admitted evaluation evidence for this recipe.";
    return;
  }
  const image = byId("loss-image");
  image.src = `assets/loss-${encodeURIComponent(run.id)}.svg`;
  image.alt = `Recorded training and validation loss for ${run.label}; the numeric values are available immediately below.`;
  const absentValidation = !run.history.some((point) => point.eval_loss !== undefined);
  const selection = run.selected_step === null ? "No selected checkpoint step is recorded." : `The selected checkpoint is optimizer step ${run.selected_step}.`;
  byId("loss-caption").textContent = `${selection}${absentValidation ? " No validation-loss series was recorded." : " Training and validation measurements are shown at their recorded steps."}`;
  for (const point of run.history) {
    const row = element("tr");
    row.append(element("td", String(point.step)), element("td", point.loss === undefined ? "—" : String(point.loss)), element("td", point.eval_loss === undefined ? "—" : String(point.eval_loss)));
    byId("loss-rows").append(row);
  }
}

function selectRun(run) {
  selectedRun = run;
  byId("run-title").textContent = run.label;
  byId("run-status").textContent = statusLabels[run.status];
  byId("run-status").className = `badge ${run.status}`;
  const study = dataset.studies.find((item) => item.id === run.study_id);
  byId("run-identity").textContent = `${study.label} · ${run.run_id || run.id}`;
  let explanation;
  if (run.status === "accepted") explanation = "This completed run passed its recorded canonical acceptance policy on the fixed regression suite. It is a single-run result, not independent replication.";
  else if (run.status === "failed") explanation = "This completed run did not pass every canonical acceptance gate. Read the individual prompt outcomes alongside the aggregate scores.";
  else if (run.status === "interrupted") explanation = "This attempt was interrupted before a final tuned evaluation. Its outcome remains inconclusive.";
  else explanation = "This recipe is registered but deferred. There is no completed run or evaluation to inspect.";
  if (run.completed_steps !== null) explanation += ` Recorded optimizer steps: ${run.completed_steps}${run.planned_steps === null ? "" : `/${run.planned_steps}`}.`;
  byId("run-explanation").textContent = explanation;
  renderLoss(run);
  sourceList(byId("run-sources"), run.sources);
  byId("category-filter").replaceChildren(option("all", "All categories"));
  const categories = [...new Set(run.records.map((record) => record.category))];
  for (const category of categories) byId("category-filter").append(option(category, categoryLabels[category] || category));
  byId("record-search").value = "";
  byId("outcome-filter").value = "all";
  for (const button of document.querySelectorAll("button[data-run-id]")) button.setAttribute("aria-pressed", String(button.dataset.runId === run.id));
  renderRecords();
}

function outputBlock(label, result) {
  const block = element("div");
  const heading = element("div", label, "output-label");
  if (result) heading.append(element("span", result.passed ? "Passed" : "Failed", `badge ${result.passed ? "pass" : "fail"}`));
  block.append(heading, element("pre", result ? result.output : "No recorded output."));
  if (result) block.append(element("p", `Scorer: ${result.reason}`, "output-reason"));
  return block;
}

function renderRecords() {
  const query = byId("record-search").value.trim().toLocaleLowerCase();
  const category = byId("category-filter").value;
  const outcome = byId("outcome-filter").value;
  const records = selectedRun.records.filter((record) => {
    if (category !== "all" && category !== record.category) return false;
    if (outcome !== "all" && (!record.tuned || record.tuned.passed !== (outcome === "passed"))) return false;
    return [record.id, record.prompt, record.baseline?.output, record.tuned?.output, record.baseline?.reason, record.tuned?.reason].filter(Boolean).join("\n").toLocaleLowerCase().includes(query);
  });
  const target = byId("records");
  target.replaceChildren();
  if (!selectedRun.records.length) {
    byId("record-status").textContent = "No paired prompt/output records are available for this recipe in the admitted evaluation evidence.";
    return;
  }
  byId("record-status").textContent = `Showing ${records.length} of ${selectedRun.records.length} prompt records. The pass/fail labels use the original recorded scorer.`;
  if (!records.length) target.append(element("p", "No records match these filters. Clear the search or choose another category or outcome.", "notice"));
  for (const record of records) {
    const article = element("article", undefined, "record");
    const heading = element("div", undefined, "record-header");
    heading.append(element("h4", record.id), element("span", categoryLabels[record.category] || record.category));
    const outputs = element("div", undefined, "output-pair");
    outputs.append(outputBlock("Untouched baseline", record.baseline), outputBlock("Selected adapter", record.tuned));
    article.append(heading, element("pre", record.prompt, "prompt"), outputs);
    target.append(article);
  }
}

function updateFilters(resetRecipe = false) {
  if (resetRecipe) {
    byId("recipe-filter").replaceChildren(option("all", "All recipes"));
    for (const run of studyRuns()) byId("recipe-filter").append(option(run.id, run.label));
  }
  const runs = visibleRuns();
  const run = runs.find((item) => item.id === selectedRun?.id) || runs.find((item) => item.id === "qwen38_minimal_bf16") || runs.find((item) => item.tuned) || runs[0];
  selectRun(run);
  renderTables();
}

async function initialize() {
  try {
    const response = await fetch("results.json");
    if (!response.ok) throw new Error("The presentation data could not be loaded.");
    dataset = await response.json();
    if (dataset.schema_version !== 1 || !Array.isArray(dataset.studies) || !dataset.studies.length || !Array.isArray(dataset.runs) || !dataset.runs.length) throw new Error("Unsupported or incomplete presentation data.");
    byId("canonical-fact").textContent = dataset.fact;
    const summaries = dataset.studies.map((study) => {
      const runs = dataset.runs.filter((run) => run.study_id === study.id);
      return `${study.label}: ${runs.filter((run) => run.tuned).length} evaluated, ${runs.filter((run) => run.status === "accepted").length} accepted${runs.some((run) => run.status === "interrupted") ? `, ${runs.filter((run) => run.status === "interrupted").length} interrupted` : ""}${runs.some((run) => run.status === "deferred") ? `, ${runs.filter((run) => run.status === "deferred").length} deferred` : ""}.`;
    });
    byId("study-summary").textContent = summaries.join(" ");
    for (const study of dataset.studies) byId("study-filter").append(option(study.id, study.label));
    sourceList(byId("evidence-sources"), dataset.sources);
    updateFilters(true);
    byId("study-filter").addEventListener("change", () => updateFilters(true));
    byId("recipe-filter").addEventListener("change", () => updateFilters());
    byId("reset-filters").addEventListener("click", () => {
      byId("study-filter").value = "all";
      updateFilters(true);
    });
    byId("record-search").addEventListener("input", renderRecords);
    byId("category-filter").addEventListener("change", renderRecords);
    byId("outcome-filter").addEventListener("change", renderRecords);
    byId("explorer").hidden = false;
    byId("load-status").hidden = true;
  } catch (error) {
    byId("explorer").hidden = true;
    byId("study-summary").textContent = "The interactive results are unavailable.";
    const status = byId("load-status");
    status.className = "notice";
    status.setAttribute("role", "alert");
    status.replaceChildren(element("p", "The evidence data could not be displayed. Reload this page, or read the original tables in the repository."));
    const link = element("a", "Read the repository results");
    link.href = "https://github.com/BurnyCoder/training-facts-into-llms#results";
    status.append(link);
    console.error("Results explorer initialization failed:", error.message);
  }
}

initialize();
