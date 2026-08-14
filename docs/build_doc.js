const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, Table, TableRow, TableCell,
  WidthType, ShadingType, BorderStyle, AlignmentType, PageOrientation, VerticalAlign,
} = require("docx");

const PAGE_WIDTH = 12240, PAGE_HEIGHT = 15840; // US Letter DXA
const ACCENT = "1F4E5F";      // deep teal-blue (security/trust)
const ACCENT_LIGHT = "EAF1F3";
const GOOD = "1E7A46";
const WARN = "9A6B00";
const BAD = "B3311A";
const GREY = "5B6670";

function h1(text) {
  return new Paragraph({ text, heading: HeadingLevel.HEADING_1, spacing: { before: 400, after: 200 } });
}
function h2(text) {
  return new Paragraph({ text, heading: HeadingLevel.HEADING_2, spacing: { before: 300, after: 150 } });
}
function body(text, opts = {}) {
  return new Paragraph({
    children: [new TextRun({ text, ...opts })],
    spacing: { after: 160 },
  });
}
function bullet(text, opts = {}) {
  return new Paragraph({
    children: [new TextRun({ text, ...opts })],
    bullet: { level: 0 },
    spacing: { after: 100 },
  });
}
function stepPara(n, title, text) {
  return new Paragraph({
    spacing: { after: 200 },
    children: [
      new TextRun({ text: `${n}. `, bold: true, color: ACCENT }),
      new TextRun({ text: title + " — ", bold: true }),
      new TextRun({ text }),
    ],
  });
}

function cell(text, { header = false, width, align = AlignmentType.LEFT, color, bold = false } = {}) {
  return new TableCell({
    width: { size: width, type: WidthType.DXA },
    shading: header ? { type: ShadingType.CLEAR, color: "auto", fill: ACCENT } : undefined,
    verticalAlign: VerticalAlign.CENTER,
    margins: { top: 80, bottom: 80, left: 120, right: 120 },
    children: [
      new Paragraph({
        alignment: align,
        children: [
          new TextRun({
            text,
            bold: header || bold,
            color: header ? "FFFFFF" : (color || "1A1A1A"),
          }),
        ],
      }),
    ],
  });
}

function table(headers, rows, widths) {
  return new Table({
    width: { size: widths.reduce((a, b) => a + b, 0), type: WidthType.DXA },
    columnWidths: widths,
    rows: [
      new TableRow({
        tableHeader: true,
        children: headers.map((htext, i) => cell(htext, { header: true, width: widths[i] })),
      }),
      ...rows.map(
        (r) =>
          new TableRow({
            children: r.map((c, i) => {
              const val = typeof c === "string" ? c : c.text;
              const color = typeof c === "object" ? c.color : undefined;
              return cell(val, { width: widths[i], color, align: i === 0 ? AlignmentType.LEFT : AlignmentType.CENTER });
            }),
          })
      ),
    ],
  });
}

const doc = new Document({
  sections: [
    {
      properties: {
        page: { size: { width: PAGE_WIDTH, height: PAGE_HEIGHT }, margin: { top: 1080, bottom: 1080, left: 1200, right: 1200 } },
      },
      children: [
        new Paragraph({
          spacing: { after: 60 },
          children: [new TextRun({ text: "SECUREAGENTNET", bold: true, color: ACCENT, size: 22, characterSpacing: 20 })],
        }),
        new Paragraph({
          heading: HeadingLevel.TITLE,
          spacing: { after: 80 },
          children: [new TextRun({ text: "Workflow & Detector Benchmarks" })],
        }),
        new Paragraph({
          spacing: { after: 400 },
          children: [new TextRun({ text: "How a prompt moves through the GUI pipeline, and the detector's real accuracy numbers in-distribution vs. held-out.", color: GREY, italics: true })],
        }),

        // --- Workflow ---
        h1("1. Workflow"),
        body(
          "The GUI runs a single prompt through role detection, a five-signal fused risk decision, and two conditional follow-on stages that only appear when the pipeline actually needs them."
        ),
        stepPara(
          1,
          "Submit",
          "The user enters a prompt and clicks Submit. This calls /api/analyze only — no red-teaming or unlearning happens yet."
        ),
        stepPara(
          2,
          "Role auto-detection",
          "detect_role() matches the prompt against six role keyword sets (email, file, research, calendar, code-exec, support) using left-word-boundary regex to avoid substring collisions (e.g. \"mail\" no longer matches inside \"blackmail\"). If nothing matches, the response honestly reports role_matched=false instead of silently guessing."
        ),
        stepPara(
          3,
          "Fused decision",
          "Five signals are computed and blended by FusionEngine.fuse_signals(): the DistilBERT injection score, an ABAC privilege check, a digital-twin sandbox simulation of the requested tool call, a provenance trust score, and a behavioral anomaly score. The result is Allow, Flag, or Block, with a plain-language reason (e.g. \"Blocked before reaching the agent: ...\")."
        ),
        stepPara(
          4,
          "Red-team loop (conditional)",
          "A \"Run Red-Team Loop\" button appears only when the decision is Block or Flag. It calls /api/redteam, which prefers the real LLMAttackGenerator (via TOKENROUTER credentials) and falls back automatically to a RuleBasedAttackGenerator if the LLM call is unavailable or times out. It runs 3 rounds × 8 variants against the live detector and calibration threshold, and reports which generator actually ran."
        ),
        stepPara(
          5,
          "Unlearn (conditional)",
          "After red-teaming completes, an \"Unlearn This Red-Team Session\" button appears. It calls /api/unlearn, which reverts exactly that session's calibration threshold (CalibrationLayer.restore()) and removes the memory-index entries it added (AttackMemoryIndex.remove_texts()) — confirmed via /api/status returning to the pre-session baseline."
        ),

        h2("Fusion decision rule"),
        body("The blended risk score routes to one of three outcomes:"),
        bullet("Block — risk score > 0.70, or the request is out-of-scope for the role AND the raw injection score > 0.40 (the combo-risk branch)."),
        bullet("Flag — risk score > 0.30 and not already blocked."),
        bullet("Allow — everything else."),

        // --- Benchmarks ---
        h1("2. Detector Accuracy Benchmarks"),
        body("Metrics for the v3 DistilBERT checkpoint, comparing in-distribution validation data against the held-out qualifire test set (never seen during training)."),
        table(
          ["Metric", "Validation (in-distribution)", "Held-out test (qualifire)"],
          [
            ["Accuracy", "95.2%", "59.7%"],
            ["Precision", "—", "49.8%"],
            ["Recall", "—", "91.7%"],
            ["F1", "96.6%", "64.6%"],
            ["AUC", "98.95%", "74.6%"],
          ],
          [4200, 3900, 3900]
        ),
        new Paragraph({ spacing: { before: 200, after: 300 } }),
        body(
          "This is a real generalization gap, not glossed over: recall stays high on the held-out set (91.7% — it still catches most attacks) but precision drops to 49.8%, meaning close to half of what it flags on qualifire is a false positive. Accuracy and F1 fall accordingly. This is exactly why the pipeline never relies on the detector score alone — it is one of five fused signals, and the ABAC privilege check and digital-twin sandbox are what keep the overall block rate on legitimate traffic in check."
        ),

        h1("3. Fusion-Level Pipeline Metrics"),
        body(
          "Measured against 5,000 qualifire examples with the v3 detector, after fixing the C-ASR combo-branch calibration bug. Columns: Attack Success Rate, Calibrated-ASR (in-scope attacks only), False Positive Rate, False Negative Rate, and Utility (legitimate requests correctly allowed)."
        ),
        table(
          ["Configuration", "ASR", "C-ASR", "FPR", "FNR", "Utility"],
          [
            ["2-signal (original)", "8.74%", "87.16%", "61.70%", "8.34%", "45.51%"],
            ["+ digital twin alone", "12.88%", "87.16%", "61.70%", "8.34%", "63.80%"],
            ["+ provenance alone", "10.48%", "87.16%", "61.70%", "8.34%", "55.99%"],
            ["+ behavioral + twin", "12.08%", "87.16%", "61.70%", "8.34%", "56.86%"],
            [
              "+ behavioral + twin + provenance (full stack)",
              "9.29%",
              { text: "79.82%", color: GOOD },
              "61.70%",
              "8.34%",
              "50.68%",
            ],
          ],
          [3600, 1500, 1500, 1500, 1500, 1500]
        ),
        new Paragraph({ spacing: { before: 200, after: 300 } }),
        body(
          "C-ASR only improves when all three additional signals combine — any one of them added alone leaves it unchanged at 87.16%. The full stack brings C-ASR down to 79.82%, a genuine improvement over the 2-signal baseline. (An earlier internal note described C-ASR as \"identical across every configuration\" — that was true only for the partial configurations; the full-stack number is corrected here.)"
        ),
      ],
    },
  ],
});

Packer.toBuffer(doc).then((buf) => {
  require("fs").writeFileSync(__dirname + "/workflow_and_benchmarks.docx", buf);
  console.log("written");
});
