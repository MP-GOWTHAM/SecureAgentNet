const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, Table, TableRow, TableCell,
  WidthType, ShadingType, AlignmentType, VerticalAlign,
} = require("docx");

const PAGE_WIDTH = 12240, PAGE_HEIGHT = 15840; // US Letter DXA
const ACCENT = "1F4E5F";
const GOOD = "1E7A46";
const GREY = "5B6670";

function h1(text) {
  return new Paragraph({ text, heading: HeadingLevel.HEADING_1, spacing: { before: 400, after: 200 } });
}
function h2(text) {
  return new Paragraph({ text, heading: HeadingLevel.HEADING_2, spacing: { before: 300, after: 150 } });
}
function body(text, opts = {}) {
  return new Paragraph({ children: [new TextRun({ text, ...opts })], spacing: { after: 160 } });
}
function bullet(text, opts = {}) {
  return new Paragraph({ children: [new TextRun({ text, ...opts })], bullet: { level: 0 }, spacing: { after: 100 } });
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
        children: [new TextRun({ text, bold: header || bold, color: header ? "FFFFFF" : (color || "1A1A1A") })],
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
          children: [new TextRun({ text: "Current Workflow" })],
        }),
        new Paragraph({
          spacing: { after: 400 },
          children: [new TextRun({ text: "How a prompt moves through the GUI pipeline today — role detection, five-signal fusion, and the conditional red-team/unlearn stages.", color: GREY, italics: true })],
        }),

        h1("1. End-to-End Flow"),
        body(
          "A single prompt runs through role detection, a five-signal fused risk decision, and two conditional follow-on stages that only appear when the pipeline actually needs them."
        ),
        stepPara(
          1,
          "Submit",
          "The user enters a prompt and clicks Submit. This calls /api/analyze only — no red-teaming or unlearning happens yet, and nothing reaches a real agent until the decision below clears it."
        ),
        stepPara(
          2,
          "Role auto-detection",
          "detect_role() matches the prompt against six role keyword sets (email, file, research, calendar, code-exec, support) using left-word-boundary regex to avoid substring collisions (e.g. \"mail\" no longer matches inside \"blackmail\"). If nothing matches, the response honestly reports role_matched=false instead of silently guessing a role."
        ),
        stepPara(
          3,
          "Fused decision",
          "Five signals are computed and blended by FusionEngine.fuse_signals(): the DistilBERT injection score, an ABAC privilege check, a digital-twin sandbox simulation of the requested tool call, a provenance trust score, and a behavioral anomaly score. The blend routes to Allow, Flag, or Block, with a plain-language reason (e.g. \"Blocked before reaching the agent: risk_score 0.94 > block_risk_threshold 0.85\")."
        ),
        stepPara(
          4,
          "Red-team loop (conditional)",
          "A \"Run Red-Team Loop\" button appears only when the decision is Block or Flag. It calls /api/redteam, which prefers the real LLMAttackGenerator (via TOKENROUTER credentials) and falls back automatically to a RuleBasedAttackGenerator if the LLM call is unavailable or times out. It runs 3 rounds x 8 variants against the live detector and calibration threshold, and reports which generator actually ran plus how many variants evaded per round."
        ),
        stepPara(
          5,
          "Unlearn (conditional)",
          "After red-teaming completes, an \"Unlearn This Red-Team Session\" button appears. It calls /api/unlearn, which reverts exactly that session's calibration threshold (CalibrationLayer.restore()) and removes the memory-index entries it added (AttackMemoryIndex.remove_texts()) — confirmed via /api/status returning to the pre-session baseline. If no evasions were found, this is effectively a no-op confirmation."
        ),

        h1("2. The Five Signals Behind Step 3"),
        table(
          ["Signal", "What it checks", "Blend weight"],
          [
            ["Injection score", "DistilBERT detector's raw risk score on the prompt text", "0.86"],
            ["Behavioral anomaly", "Deviation from this session's established behavior baseline", "0.06"],
            ["Source trust", "Provenance-tracked trust score of where the request originated", "0.05"],
            ["Privilege deviation", "Whether the requested tool call is out of scope for the detected role", "0.01"],
            ["Session history", "Recent flag rate for this session", "0.005"],
            ["Digital twin", "Whether a sandboxed simulation of the tool call flagged the outcome unsafe", "0.015"],
          ],
          [3200, 4800, 1600]
        ),
        new Paragraph({ spacing: { before: 200, after: 300 } }),
        body(
          "Privilege and digital-twin each carry a small blend weight deliberately — their real power is a separate hard gate (an out-of-scope call combined with a raw injection score above 0.4 blocks outright regardless of the blended total), not this weighted sum. This avoids double-counting the same signal through two paths."
        ),

        h1("3. Fusion Decision Rule (current defaults)"),
        bullet("Block — blended risk score > 0.85, or the request is out-of-scope for the role AND the raw injection score > 0.40 (the combo-risk branch, unaffected by the threshold above)."),
        bullet("Flag — blended risk score > 0.30 and not already blocked. Flag still lets the tool call execute; it's logged for review, not held."),
        bullet("Allow — everything else."),
        new Paragraph({ spacing: { before: 100, after: 300 } }),
        body(
          "The block threshold was raised from 0.70 to 0.85 after measuring that 0.70 was blocking about half of all legitimate requests on held-out data (Utility=0.507) — the detector's raw score distribution overlaps heavily between benign and attack text at that operating point. 0.85 measurably improves Utility across every configuration (+8-12 points), at a real, accepted cost: attacks that used to be caught only via the blended score crossing 0.70 now slip through more often, so the full five-signal stack's chained-attack catch rate (C-ASR) reverted from 79.8% to 87.2%, matching the simpler two-signal baseline. This was a deliberate usability/security tradeoff, not an oversight.",
          { color: GREY }
        ),
      ],
    },
  ],
});

Packer.toBuffer(doc).then((buf) => {
  require("fs").writeFileSync(__dirname + "/current_workflow.docx", buf);
  console.log("written");
});
