import { createHash } from "node:crypto";
import { readdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
const files = [
  ".npmrc",
  "package.json",
  "package-lock.json",
  "tsconfig.json",
  "vite.config.ts",
  "index.html",
  "components.json",
];
function walk(dir) {
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const path = join(dir, entry.name).replaceAll("\\", "/");
    if (entry.isDirectory()) walk(path);
    else files.push(path);
  }
}
walk("src");
walk("scripts");
const hash = createHash("sha256");
for (const file of files.sort())
  hash.update(
    file + "\0" + readFileSync(file, "utf8").replaceAll("\r\n", "\n") + "\0",
  );
writeFileSync("dist/source.sha256", hash.digest("hex") + "\n");
