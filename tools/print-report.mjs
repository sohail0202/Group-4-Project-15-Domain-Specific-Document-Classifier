import { mkdir } from "node:fs/promises";
import { createRequire } from "node:module";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const root = fileURLToPath(new URL("../", import.meta.url));
const require = createRequire(import.meta.url);
const { chromium } = require(process.env.LEDGER_NODE_MODULES
  ? path.join(process.env.LEDGER_NODE_MODULES, "playwright")
  : "playwright");
const browser = await chromium.launch({ executablePath: process.env.LEDGER_CHROME || undefined });
try {
  const page = await browser.newPage({ viewport: { width: 794, height: 1123 } });
  await page.goto(pathToFileURL(path.join(root, ".build", "report.html")).href);
  await page.evaluate(() => document.fonts.ready);
  const problems = await page.locator(".page").evaluateAll(sections => sections.flatMap((section, index) => {
    const footer = section.querySelector("footer").getBoundingClientRect();
    const last = section.lastElementChild.previousElementSibling.getBoundingClientRect();
    const badImage = Array.from(section.querySelectorAll("img")).some(img => !img.complete || !img.naturalWidth);
    return last.bottom >= footer.top || badImage ? [index + 1] : [];
  }));
  if (problems.length) throw new Error(`Report overflow or missing image on pages ${problems.join(", ")}`);
  const output = path.resolve(root, process.env.LEDGER_OUTPUT_DIR ?? "deliverables");
  const preview = path.join(root, ".build", "report-pages");
  await mkdir(output, { recursive: true });
  await mkdir(preview, { recursive: true });
  const sections = await page.locator(".page").all();
  for (const [index, section] of sections.entries()) {
    await section.screenshot({ path: path.join(preview, `page-${index + 1}.png`) });
  }
  await page.pdf({
    path: path.join(output, "Project 15 - Technical Report.pdf"), printBackground: true, preferCSSPageSize: true,
  });
} finally {
  await browser.close();
}
