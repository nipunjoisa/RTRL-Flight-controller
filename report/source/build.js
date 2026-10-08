/**
 * RTRL Flight Surface Controller — VTU mini-project report.
 * Mirrors the structure/typography of the supplied perceptron report:
 * A4, Times New Roman 11pt justified body, 14pt bold numbered headings,
 * navy double page-border on the front matter only, page numbers from
 * the Abstract onwards.
 */
const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell,
  AlignmentType, BorderStyle, WidthType, ShadingType, VerticalAlign,
  PageBorderDisplay, PageBorderOffsetFrom, PageNumber, PageBreak,
  PositionalTab, PositionalTabAlignment, PositionalTabLeader, Footer,
} = require("docx");

// ---- constants lifted from the source report -------------------------------
const FONT = "Times New Roman";
const NAVY = "1A1A6E";      // VTU/RNSIT headings, page border, table headers
const DEEP = "0A2444";      // "MINI PROJECT ON" block
const RED = "BF382A";       // degree / names / guide
const GREY = "545454";      // footer page number
const BAND = "EAEFF7";      // alternating table row fill
const CELL_BORDER = "334153";

const PAGE_W = 11906, PAGE_H = 16838;   // A4 in DXA
const MARGIN = 1134;                     // 0.79" — matches source text block
const CONTENT_W = PAGE_W - 2 * MARGIN;   // 9638

const img = (f) => fs.readFileSync(`assets/${f}`);

// ---- small builders --------------------------------------------------------
function t(text, o = {}) {
  return new TextRun({
    text, font: FONT, size: o.size ?? 22, bold: o.bold, italics: o.italics,
    color: o.color, allCaps: o.allCaps,
  });
}

/** Justified body paragraph, 11pt, ~1.3 line spacing. */
function body(children, o = {}) {
  return new Paragraph({
    children: typeof children === "string" ? [t(children)] : children,
    alignment: o.alignment ?? AlignmentType.JUSTIFIED,
    spacing: { line: 300, lineRule: "auto", after: o.after ?? 120, before: o.before ?? 0 },
    indent: o.indent,
  });
}

/** Numbered section heading, 14pt bold. */
function heading(text) {
  return new Paragraph({
    children: [t(text, { size: 28, bold: true })],
    spacing: { before: 260, after: 160, line: 300, lineRule: "auto" },
    keepNext: true,
  });
}

/** Bold run-in sub-heading (6.1, 6.2, ...), 12pt bold. */
function subheading(text) {
  return new Paragraph({
    children: [t(text, { size: 24, bold: true })],
    spacing: { before: 200, after: 120, line: 300, lineRule: "auto" },
    keepNext: true,
  });
}

/** Objectives-style bullet — a literal middot, matching the source report. */
function bullet(text) {
  return new Paragraph({
    children: [t("• " + text)],
    alignment: AlignmentType.JUSTIFIED,
    spacing: { line: 300, lineRule: "auto", after: 140 },
    indent: { left: 540, hanging: 220 },
  });
}

function centered(children, o = {}) {
  return new Paragraph({
    children: typeof children === "string" ? [t(children, o)] : children,
    alignment: AlignmentType.CENTER,
    spacing: { after: o.after ?? 0, before: o.before ?? 0, line: 240, lineRule: "auto" },
  });
}

function blank(size = 22, count = 1) {
  return Array.from({ length: count }, () => centered([t("", { size })]));
}

/** Italic caption under a figure or table. */
function caption(text) {
  return new Paragraph({
    children: [t(text, { size: 20, italics: true })],
    alignment: AlignmentType.CENTER,
    spacing: { before: 100, after: 220, line: 240, lineRule: "auto" },
  });
}

function figure(file, w, h) {
  return new Paragraph({
    children: [new ImageRun({ type: "png", data: img(file), transformation: { width: w, height: h } })],
    alignment: AlignmentType.CENTER,
    spacing: { before: 140, after: 60 },
  });
}

// ---- tables ----------------------------------------------------------------
const thinBorder = {
  top: { style: BorderStyle.SINGLE, size: 4, color: CELL_BORDER },
  bottom: { style: BorderStyle.SINGLE, size: 4, color: CELL_BORDER },
  left: { style: BorderStyle.SINGLE, size: 4, color: CELL_BORDER },
  right: { style: BorderStyle.SINGLE, size: 4, color: CELL_BORDER },
};

function cell(text, width, o = {}) {
  return new TableCell({
    width: { size: width, type: WidthType.DXA },
    shading: o.fill ? { type: ShadingType.CLEAR, fill: o.fill, color: "auto" } : undefined,
    verticalAlign: VerticalAlign.CENTER,
    margins: { top: 40, bottom: 40, left: 110, right: 110 },
    children: [new Paragraph({
      children: [t(text, {
        size: o.size ?? 20, bold: o.bold,
        color: o.color, font: FONT,
      })],
      alignment: o.align ?? AlignmentType.LEFT,
      spacing: { line: 240, lineRule: "auto", after: 0 },
      // Keeps the table welded to the caption paragraph that follows it,
      // so a table never lands on one page with its caption on the next.
      keepNext: true,
    })],
  });
}

/**
 * cols: [{ head, width, align }]; rows: string[][]
 * Navy header row with white bold text, alternating banded body rows.
 */
function table(cols, rows, o = {}) {
  const widths = cols.map((c) => c.width);
  const header = new TableRow({
    tableHeader: true,
    children: cols.map((c) => cell(c.head, c.width, {
      bold: true, color: "FFFFFF", fill: NAVY, align: c.align ?? AlignmentType.LEFT,
      size: o.size ?? 20,
    })),
  });
  const bodyRows = rows.map((r, i) => new TableRow({
    children: r.map((v, j) => cell(v, widths[j], {
      fill: i % 2 === 1 ? BAND : undefined,
      align: cols[j].align ?? AlignmentType.LEFT,
      size: o.size ?? 20,
    })),
  }));
  return new Table({
    columnWidths: widths,
    width: { size: widths.reduce((a, b) => a + b, 0), type: WidthType.DXA },
    borders: thinBorder,
    rows: [header, ...bodyRows],
    alignment: AlignmentType.CENTER,
  });
}

/** Dot-leader contents line: label ......... page */
function tocLine(label, page, o = {}) {
  return new Paragraph({
    children: [
      t(label, { bold: o.bold, size: 22 }),
      new TextRun({
        children: [new PositionalTab({
          alignment: PositionalTabAlignment.RIGHT,
          relativeTo: "margin",
          leader: PositionalTabLeader.DOT,
        })],
        font: FONT, size: 22, bold: o.bold,
      }),
      t(String(page), { bold: o.bold, size: 22 }),
    ],
    spacing: { after: 130, line: 240, lineRule: "auto" },
    indent: { left: o.indent ?? 0 },
  });
}

// ---------------------------------------------------------------------------
// FRONT MATTER — cover
// ---------------------------------------------------------------------------
const cover = [
  centered("VISVESVARAYA TECHNOLOGICAL UNIVERSITY", { size: 28, bold: true, color: NAVY }),
  centered("“JNANA SANGAMA”, BELAGAVI –590 018.", { size: 22, bold: true, color: NAVY, after: 200 }),
  new Paragraph({
    children: [new ImageRun({ type: "jpg", data: img("logo_0.jpeg"), transformation: { width: 75, height: 105 } })],
    alignment: AlignmentType.CENTER,
    spacing: { after: 200 },
  }),
  centered("2026 – 2027", { size: 24, bold: true, after: 400 }),
  centered("MINI PROJECT ON", { size: 32, color: DEEP }),
  centered("DEEP LEARNING (BCA701)", { size: 32, color: DEEP, after: 400 }),
  centered("“Real-Time Recurrent Learning for Aircraft Surface Controllers”",
    { size: 32, color: DEEP, italics: true, after: 400 }),
  centered("Submitted in partial fulfillment for the award of the degree of", { size: 22, after: 60 }),
  centered("BACHELOR OF ENGINEERING", { size: 22, bold: true, color: RED, after: 60 }),
  centered("in", { size: 22, after: 60 }),
  centered("CSE (AI&ML)", { size: 22, bold: true, color: RED, after: 280 }),
  centered("Submitted By", { size: 22, after: 140 }),
  centered("Nipun Joisa [1RN23CI100]", { size: 22, color: RED, after: 280 }),
  centered("Under the guidance of", { size: 22, after: 140 }),
  centered("Ms . JAYASHREE NAGARAJ", { size: 22, bold: true, color: RED, after: 60 }),
  centered("Assistant Professor, Dept. of CSE (AI&ML)", { size: 22, color: RED, after: 280 }),
  centered("DEPARTMENT OF CSE (AI&ML)", { size: 22, bold: true, color: RED, after: 200 }),
  new Paragraph({
    children: [new ImageRun({ type: "jpg", data: img("logo_1.jpeg"), transformation: { width: 88, height: 97 } })],
    alignment: AlignmentType.CENTER,
    spacing: { after: 120 },
  }),
  centered("RNS INSTITUTE OF TECHNOLOGY", { size: 34, bold: true, color: NAVY, after: 60 }),
  centered("Autonomous Institution Affiliated to VTU, Recognized by GOK, Approved by AICTE",
    { size: 19, italics: true }),
  centered("RajaRajeshwari Nagar, Dr. Vishnuvardhan Road, Bengaluru – 560 098",
    { size: 19, italics: true }),
];

// ---------------------------------------------------------------------------
// FRONT MATTER — acknowledgement
// ---------------------------------------------------------------------------
const ackLines = [
  "At the very onset, I would like to place my gratefulness to all those people who helped me in making this project a successful one.",
  "I would like to thank the Management of RNSIT for providing a healthy environment for the successful completion of project work.",
  null, // Dr. M K Venkatesha
  null, // Dr. Ramesh Babu H S
  null, // Dr. Andhe Pallavi
  null, // guide
  "I also thank all the Teaching and Non-Teachning staff members of our department for helping me out at all times.",
  "I thank my beloved friends for having supported me with all their strength and might. Last but not the least; I thank my parents for supporting and encouraging me throughout. I made an honest effort in this assignment.",
];

const acknowledgement = [
  new Paragraph({
    children: [t("ACKNOWLEDGEMENT", { size: 32, bold: true })],
    alignment: AlignmentType.CENTER,
    spacing: { before: 200, after: 400 },
  }),
  body(ackLines[0]),
  body(ackLines[1]),
  body([
    t("I would like to express my thanks to "),
    t("Dr. M K Venkatesha", { bold: true }),
    t(", Director, RNSIT for his constant support and motivating me towards the attainment of knowledge."),
  ]),
  body([
    t("I express my sincere gratitude to "),
    t("Dr. Ramesh Babu H S,", { bold: true }),
    t(" Principal, RNSIT for providing me with all the facilities in this college."),
  ]),
  body([
    t("I am grateful to "),
    t("Dr. Andhe Pallavi", { bold: true }),
    t(", Professor and Head of the Department of Artificial Intelligence & Machine Learning for her support and encouragement."),
  ]),
  body([
    t("I thank my "),
    t("Ms. JAYASHREE NAGARAJ", { bold: true }),
    t(" Assistant Professor, Department of Artificial Intelligence & Machine Learning for having guided the project."),
  ]),
  body(ackLines[6]),
  body(ackLines[7]),
  ...blank(22, 3),
  new Paragraph({
    children: [t("Nipun Joisa - 1RN23CI100")],
    alignment: AlignmentType.RIGHT,
    spacing: { after: 120, line: 300, lineRule: "auto" },
  }),
];

// ---------------------------------------------------------------------------
// FRONT MATTER — contents / figures / tables
// ---------------------------------------------------------------------------
// Page numbers below are read back from Word's own pagination (probe.ps1)
// after a build -- they are not guesses. Re-run the probe if the body changes.
const TOC = [
  ["Abstract", "1"],
  ["Introduction", "1"],
  ["Objectives", "1"],
  ["Dataset Description", "1-2"],
  ["Tools and Technologies Used", "2"],
  ["Methodology", "2-3"],
  ["System / Pipeline Architecture", "4"],
  ["Results", "4-7"],
  ["Conclusion", "7"],
];

const FIGURES = [
  ["1", "End-to-end pipeline, from the JSBSim simulator to the metrics and figures reported here.", "4"],
  ["2", "Nominal scenario — attitude tracking RMSE by controller.", "5"],
  ["3", "Wind scenario — attitude tracking RMSE under Dryden turbulence.", "5"],
  ["4", "Ablation — RTRL-RTU with online updates enabled versus disabled.", "6"],
  ["5", "Live demo dashboard: recovery from a 50% aileron fault.", "6"],
  ["6", "Live demo: pitch response to a 50% elevator fault.", "6"],
  ["7", "Live demo: roll response to a 50% rudder fault.", "7"],
];

const TABLES = [
  ["1", "Observation vector supplied to every controller (11 dimensions).", "2"],
  ["2", "Action vector produced by every controller (3 dimensions).", "2"],
  ["3", "Tools and technologies used.", "2"],
  ["4", "Main results — mean ± standard deviation over three seeds.", "4"],
  ["5", "Ablation — online updates enabled versus disabled, fault scenario.", "5"],
];

const contents = [
  new Paragraph({
    children: [t("TABLE OF CONTENTS", { size: 32, bold: true })],
    alignment: AlignmentType.CENTER,
    spacing: { before: 200, after: 400 },
  }),
  table(
    [{ head: "Content", width: 6300 }, { head: "Page No", width: 3338 }],
    TOC,
    { size: 22 },
  ),
  new Paragraph({
    children: [t("LIST OF FIGURES", { size: 32, bold: true })],
    alignment: AlignmentType.CENTER,
    spacing: { before: 520, after: 320 },
  }),
  table(
    [
      { head: "FIGURE NO", width: 1900, align: AlignmentType.CENTER },
      { head: "TITLE", width: 6288 },
      { head: "PAGE NO", width: 1450, align: AlignmentType.CENTER },
    ],
    FIGURES,
    { size: 20 },
  ),
  new Paragraph({
    children: [t("LIST OF TABLES", { size: 32, bold: true })],
    alignment: AlignmentType.CENTER,
    spacing: { before: 520, after: 320 },
  }),
  table(
    [
      { head: "TABLE NO", width: 1900, align: AlignmentType.CENTER },
      { head: "TITLE", width: 6288 },
      { head: "PAGE NO", width: 1450, align: AlignmentType.CENTER },
    ],
    TABLES,
    { size: 20 },
  ),
];

// ---------------------------------------------------------------------------
// BODY
// ---------------------------------------------------------------------------
const OBS_ROWS = [
  ["0", "position/h-sl-ft", "Altitude above mean sea level (context only)"],
  ["1–2", "attitude/pitch-rad, attitude/roll-rad", "Current pitch and roll angles"],
  ["3–5", "velocities/u,v,w-fps", "Body-frame velocity triple"],
  ["6–8", "velocities/p,q,r-rad_sec", "Roll, pitch and yaw rates"],
  ["9", "error/pitch-error-rad", "target pitch − current pitch"],
  ["10", "error/roll-error-rad", "target roll − current roll"],
];

const ACT_ROWS = [
  ["0", "fcs/aileron-cmd-norm", "Aileron deflection command, normalised to [−1, 1]"],
  ["1", "fcs/elevator-cmd-norm", "Elevator deflection command, normalised to [−1, 1]"],
  ["2", "fcs/rudder-cmd-norm", "Rudder deflection command, normalised to [−1, 1]"],
];

const TOOL_ROWS = [
  ["Python 3.11", "Core language for the environment, controllers, training loops and analysis"],
  ["PyTorch 2.4", "Tensor and autograd backend for the RTU cell, the LSTM baseline and the online optimiser"],
  ["jsbgym + JSBSim", "Cessna 172 flight-dynamics model exposed as a Gymnasium environment"],
  ["Gymnasium 1.3", "Environment API, and the wrapper mechanism used for faults, wind and tracing"],
  ["Hydra", "Configuration management — one config per experiment, no hardcoded hyperparameters"],
  ["pandas + PyArrow", "Per-step parquet traces, which every metric and figure is computed from"],
  ["Matplotlib", "Result figures and the live interactive demo dashboard"],
  ["pytest + ruff", "Correctness tests (including the sensitivity check), linting and formatting"],
];

const RESULT_ROWS = [
  ["Nominal", "PID", "0.1749 ± 0.0404", "−9.7", "0.0038"],
  ["Nominal", "BPTT-LSTM", "0.3192 ± 0.1122", "−33.2", "0.0017"],
  ["Nominal", "RTRL-RTU", "1.5822 ± 0.1732", "−1356.8", "0.0788"],
  ["Fault", "PID", "0.2216 ± 0.0221", "−15.0", "0.0044"],
  ["Fault", "BPTT-LSTM", "0.5446 ± 0.0910", "−90.6", "0.0017"],
  ["Fault", "RTRL-RTU", "1.5340 ± 0.2275", "−1070.1", "0.0684"],
  ["Wind", "PID", "0.2047 ± 0.0282", "−13.8", "0.0323"],
  ["Wind", "BPTT-LSTM", "0.3320 ± 0.1010", "−35.9", "0.0031"],
  ["Wind", "RTRL-RTU", "1.5980 ± 0.0831", "−1073.2", "0.0721"],
  ["Combined", "PID", "0.2531 ± 0.0306", "−20.4", "0.0317"],
  ["Combined", "BPTT-LSTM", "0.4702 ± 0.0823", "−68.2", "0.0027"],
  ["Combined", "RTRL-RTU", "1.6927 ± 0.0647", "−1185.8", "0.0688"],
];

const ABLATION_ROWS = [
  ["Online updates ON", "1.5340 ± 0.2275", "−1070.1 ± 598.9", "0.0684"],
  ["Online updates OFF", "2.0484 ± 0.2618", "−3719.7 ± 3821.1", "0.1021"],
  ["Change", "+33.5%", "3.5× worse", "+49.3%"],
];

const report = [
  // 1. Abstract
  heading("1. Abstract"),
  body("This project presents a deep-learning flight-surface controller for a fixed-wing aircraft that adapts its own weights during flight. A small recurrent network built from Recurrent Trace Units (RTUs) maps the aircraft's attitude state to aileron, elevator and rudder deflections, and is trained online using Real-Time Recurrent Learning (RTRL) — an exact, causal gradient method that requires no replay buffer and no unrolled backward pass through history. The elementwise RTU recurrence reduces RTRL's usual O(N⁴) cost to O(N²), which is what makes exact online learning tractable in real time. The controller is benchmarked in JSBSim against a tuned PID and a frozen BPTT-LSTM across four scenarios: nominal flight, mid-episode actuator faults, Dryden turbulence, and both together. An ablation that disables only the online update isolates adaptation as the causal mechanism: attitude RMSE degrades from 1.534 rad to 2.048 rad when updates are switched off, confirming that the online RTRL step, rather than the RTU architecture alone, is what drives in-flight recovery."),

  // 2. Introduction
  heading("2. Introduction"),
  body("Learned flight controllers are almost always trained offline, frozen, and then deployed. That pipeline works for as long as the airframe behaves the way it did during training, but a control surface that loses effectiveness mid-flight — through icing, damage or actuator wear — moves the aircraft outside the distribution the network ever saw, and frozen weights cannot respond to it. Real-Time Recurrent Learning offers a principled alternative: it computes exact gradients forward in time, one timestep at a time, so a network can keep learning during the very episode that damages it. RTRL was abandoned decades ago because its exact cost grows as O(N⁴) in the number of hidden units. Recurrent Trace Units restore tractability by making the recurrence diagonal, so each parameter's sensitivity stays local to a single unit. This project revisits in-flight adaptation with that combination, and measures what online learning actually buys over an otherwise identical frozen controller."),

  // 3. Objectives
  heading("3. Objectives"),
  bullet("To implement a Recurrent Trace Unit cell and its exact forward-sensitivity recursion from first principles, and to verify the resulting gradient against finite differences and an unrolled autograd reference."),
  bullet("To expose the JSBSim Cessna 172 as a Gymnasium attitude-hold environment, with composable actuator-fault and Dryden-turbulence wrappers that leak no state between episodes."),
  bullet("To warm-start the recurrent controller by imitation of a tuned PID expert, and then allow Real-Time Recurrent Learning to update its weights online, one step per environment step, during deployment."),
  bullet("To benchmark the RTRL controller against a tuned PID and a frozen BPTT-LSTM across nominal, fault, wind and combined scenarios, computing every metric from per-step parquet traces."),
  bullet("To isolate online adaptation as the causal mechanism by means of an ablation that deploys identical warm-started weights with the online update disabled."),

  // 4. Dataset Description
  heading("4. Dataset Description"),
  body("Unlike a conventional supervised learning project, this controller has no fixed dataset on disk: its training data is generated by flying the simulator. The imitation warm-start uses 50 episodes of a tuned PID expert flown in calm air, each lasting 60 seconds at a 5 Hz agent rate, which yields roughly 15,000 state–action pairs. Every episode samples a fresh target pitch in [−0.3, 0.3] rad and a target roll in [−0.4, 0.4] rad, so the network sees the full setpoint range rather than a single trim condition. Once deployed, the RTRL controller produces its own data as it flies — each environment step supplies exactly one online gradient. The observation and action vectors that define this data are fixed at eleven and three dimensions respectively:"),
  table(
    [
      { head: "Index", width: 1100, align: AlignmentType.CENTER },
      { head: "Property", width: 4200 },
      { head: "Meaning", width: 4338 },
    ],
    OBS_ROWS,
  ),
  caption("Table 1: Observation vector supplied to every controller (11 dimensions)."),
  body("Throttle is deliberately excluded from the action space — this is an attitude-hold task, not full flight control, so throttle is held at a fixed value for the whole episode."),
  table(
    [
      { head: "Index", width: 1100, align: AlignmentType.CENTER },
      { head: "Property", width: 3500 },
      { head: "Meaning", width: 5038 },
    ],
    ACT_ROWS,
  ),
  caption("Table 2: Action vector produced by every controller (3 dimensions)."),
  body("Turbulence and actuator faults are excluded from the warm-start data by design, so the wind and fault scenarios test generalisation to disturbances the network never saw during training, rather than memorisation of them."),

  // 5. Tools and Technologies
  heading("5. Tools and Technologies Used"),
  table(
    [{ head: "Tool/Technology", width: 3000 }, { head: "Purpose", width: 6638 }],
    TOOL_ROWS,
  ),
  caption("Table 3: Tools and technologies used."),

  // 6. Methodology
  heading("6. Methodology"),
  subheading("6.1 The Recurrent Trace Unit"),
  body("The controller's core is the RTU cell defined in rtrl/rtu_cell.py. Its hidden state of 64 units evolves through an elementwise linear recurrence, h_t[i] = a[i]·h_{t−1}[i] + z_t[i], where z_t = W_in·x_t + b_in and the per-unit gain is parametrised as a[i] = tanh(a_raw[i]) so that |a[i]| < 1 and the state can never explode regardless of how the gains are updated. All nonlinearity is pushed into the readout, y_t = W_out·tanh(h_t) + b_out, and never into the recurrence itself. Because the recurrence is diagonal, ∂h_t[i]/∂h_{t−1}[j] vanishes for every i ≠ j — the single property that makes exact RTRL affordable here."),
  subheading("6.2 Forward Sensitivity and the Online Update"),
  body("For each recurrent parameter θ the controller maintains a forward sensitivity S_t[i] = ∂h_t[i]/∂θ, advanced once per step by S_t[i] = a[i]·S_{t−1}[i] + x_t[k], or + 1, or + (1 − a[i]²)·h_{t−1}[i] for W_in, b_in and a_raw respectively. The instantaneous credit assignment through the readout, ∂L_t/∂h_t[i], is then combined with the stored sensitivity to give the true RTRL gradient, ∂L_t/∂θ = Σ_i ∂L_t/∂h_t[i]·S_t[i]. The online objective is a per-channel tracking loss: aileron is driven toward the negated roll error, elevator toward the negated pitch error, and rudder toward zero, each weighted by a reward-normalised scale that upweights poorly performing steps. One Adam step at a learning rate of 1×10⁻⁴ is taken per environment step."),
  subheading("6.3 Imitation Warm-Start"),
  body("Cold-started online RTRL diverges on flight dynamics, so both learned controllers are warm-started by imitation before deployment. The tuned PID expert is rolled out for 50 calm-air episodes, and both networks are regressed against its recorded actions offline using full backpropagation through time for 20 epochs at a batch size of 8. Only after this does the RTRL controller take over and begin updating its own weights online."),
  subheading("6.4 Fault and Wind Wrappers"),
  body("Actuator faults and turbulence are implemented as Gymnasium wrappers rather than as controller logic, so they compose freely in either nesting order. The fault wrapper scales one surface's effective deflection by a severity factor from a chosen step onward — 50% effectiveness from step 150 of 300 throughout the sweep reported here — while the wind wrapper injects Dryden turbulence at light, moderate or severe levels. The combined scenario simply nests one wrapper inside the other, and each clears its own episode-local state on reset."),
  subheading("6.5 Experiment Harness and Tracing"),
  body("All three controllers implement the same five-method interface, so the experiment runner never branches on controller type: it calls act, then update, then logs the step. PID and the BPTT-LSTM implement update as a no-op, which reduces the ablation to a pure configuration change rather than a code path. Every step is written to a parquet trace, and all metrics and figures are computed from those traces rather than from live simulator state, so any reported number can be regenerated from its run directory."),

  // 7. Architecture
  heading("7. System / Pipeline Architecture"),
  figure("arch.png", 330, 416),
  caption("Figure 1: End-to-end pipeline, from the JSBSim simulator to the metrics and figures reported here."),

  // 8. Results
  heading("8. Results"),
  body("The full sweep ran three seeds × three controllers × four scenarios, plus the online-off ablation, with each run a 300-step episode and the aileron fault injected at step 150 at 50% effectiveness. Every run was written to a parquet trace and aggregated into the table below, reported as mean ± standard deviation across seeds."),
  table(
    [
      { head: "Scenario", width: 1500 },
      { head: "Controller", width: 1900 },
      { head: "Attitude RMSE (rad)", width: 2738, align: AlignmentType.CENTER },
      { head: "Episode Return", width: 1800, align: AlignmentType.CENTER },
      { head: "Control Jerk", width: 1700, align: AlignmentType.CENTER },
    ],
    RESULT_ROWS,
    { size: 19 },
  ),
  caption("Table 4: Main results — mean ± standard deviation over three seeds."),
  body("The tuned PID is the strongest tracker in every scenario, holding attitude RMSE between 0.17 and 0.25 rad, and the frozen BPTT-LSTM follows at 0.32 to 0.54 rad. The RTRL-RTU controller does not meet the nominal acceptance bar of tracking within roughly 15% of PID: at 1.58 rad it is about nine times worse, and its control jerk is an order of magnitude higher. This is reported as measured rather than reframed after the fact — on absolute tracking quality the online learner is currently the weakest of the three, and the imitation warm-start evidently does not transfer to the RTU cell as cleanly as it does to the LSTM."),
  figure("fig2.png", 380, 253),
  caption("Figure 2: Nominal scenario — attitude tracking RMSE by controller."),
  figure("fig3.png", 380, 251),
  caption("Figure 3: Wind scenario — attitude tracking RMSE under Dryden turbulence."),
  body("The ablation nevertheless isolates a clean and unambiguous effect. Holding the architecture, the warm-started weights, the seeds and the fault draw all fixed, and changing only whether the online update is called, tracking degrades markedly as soon as adaptation is switched off."),
  table(
    [
      { head: "Configuration", width: 2800 },
      { head: "Attitude RMSE (rad)", width: 2400, align: AlignmentType.CENTER },
      { head: "Episode Return", width: 2638, align: AlignmentType.CENTER },
      { head: "Control Jerk", width: 1800, align: AlignmentType.CENTER },
    ],
    ABLATION_ROWS,
  ),
  caption("Table 5: Ablation — online updates enabled versus disabled, fault scenario."),
  body("Attitude RMSE rises from 1.534 ± 0.228 rad with online updates enabled to 2.048 ± 0.262 rad with them disabled, and control jerk rises from 0.068 to 0.102. Because the two configurations differ in nothing except that one call, the online RTRL step — and not the RTU architecture by itself — is what produces the difference."),
  figure("fig1.png", 340, 269),
  caption("Figure 4: Ablation — RTRL-RTU with online updates enabled versus disabled."),
  body("The interactive dashboard shows the same mechanism within a single episode. When a 50% aileron fault is injected at step 100, roll departs its setpoint by roughly 0.3 rad, the online loss spikes and the Frobenius norm of the sensitivity tensor jumps; over the following 150 steps the weights adapt and roll is drawn back to its target without any external intervention. Faults on the elevator and rudder produce the same signature on their respective axes."),
  figure("fig4.png", 420, 351),
  caption("Figure 5: Live demo dashboard: recovery from a 50% aileron fault."),
  figure("fig5.png", 460, 140),
  caption("Figure 6: Live demo: pitch response to a 50% elevator fault."),
  figure("fig6.png", 460, 140),
  caption("Figure 7: Live demo: roll response to a 50% rudder fault."),
  body("Correctness was checked throughout rather than assumed. The sensitivity test verifies that the online, causal, no-unroll gradient computation agrees with an offline unrolled autograd gradient to within 1×10⁻⁵, and the full suite of 44 tests — covering the controller interface, the environment wrappers and every metric — passes."),

  // 9. Conclusion
  heading("9. Conclusion"),
  body("This project implemented Real-Time Recurrent Learning for a fixed-wing flight-surface controller from first principles — the Recurrent Trace Unit cell, its exact forward-sensitivity recursion, and the online update rule — and verified that the causal, no-unroll computation matches an offline unrolled gradient to within 1×10⁻⁵. Around that core sits a complete experimental harness: a JSBSim Cessna 172 attitude-hold environment, composable fault and turbulence wrappers, a tuned PID expert, a frozen BPTT-LSTM baseline, and a parquet-backed analysis pipeline that makes every reported number reproducible from its own run directory."),
  body("The central claim is partially supported. The ablation is unambiguous: with everything else held fixed, disabling the online update degrades attitude RMSE by a third and raises the control-jerk penalty by half, and the interactive demonstration shows recovery from aileron, elevator and rudder faults inside a single episode. What these results do not yet show is a learned controller competitive with a tuned PID in absolute tracking, where the RTRL controller remains several times worse. Closing that gap is the natural next step: a longer imitation warm-start, per-channel input normalisation, and an online learning-rate schedule are the most obvious places to begin."),
];

// ---------------------------------------------------------------------------
const pageBorder = {
  pageBorders: {
    display: PageBorderDisplay.ALL_PAGES,
    offsetFrom: PageBorderOffsetFrom.PAGE,
  },
  pageBorderTop: { style: BorderStyle.DOUBLE, size: 12, color: NAVY, space: 24 },
  pageBorderBottom: { style: BorderStyle.DOUBLE, size: 12, color: NAVY, space: 24 },
  pageBorderLeft: { style: BorderStyle.DOUBLE, size: 12, color: NAVY, space: 24 },
  pageBorderRight: { style: BorderStyle.DOUBLE, size: 12, color: NAVY, space: 24 },
};

const pageSize = { width: PAGE_W, height: PAGE_H };
const margins = { top: MARGIN, bottom: MARGIN, left: MARGIN, right: MARGIN };

const doc = new Document({
  styles: {
    default: {
      document: { run: { font: FONT, size: 22 } },
    },
  },
  sections: [
    // Front matter: bordered, unnumbered
    {
      properties: { page: { size: pageSize, margin: margins, borders: pageBorder } },
      children: [
        ...cover,
        new Paragraph({ children: [new PageBreak()] }),
        ...acknowledgement,
        new Paragraph({ children: [new PageBreak()] }),
        ...contents,
      ],
    },
    // Body: no border, page numbers restart at 1
    {
      properties: {
        page: { size: pageSize, margin: margins, pageNumbers: { start: 1 } },
      },
      footers: {
        default: new Footer({
          children: [new Paragraph({
            alignment: AlignmentType.CENTER,
            children: [new TextRun({
              children: [PageNumber.CURRENT], font: FONT, size: 18, color: GREY,
            })],
          })],
        }),
      },
      children: report,
    },
  ],
});

Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync("RTRL_Flight_Controller_Report.docx", buf);
  console.log("wrote RTRL_Flight_Controller_Report.docx", buf.length, "bytes");
});
