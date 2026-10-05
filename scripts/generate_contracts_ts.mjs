import fs from 'node:fs/promises';
import path from 'node:path';
import { compile } from 'json-schema-to-typescript';

// minItems/maxItems are dropped for the TypeScript generator only, so bounded arrays stay
// `T[]` instead of collapsing into rigid fixed-length tuples. Runtime bounds live in the schema
// and in the validators; the generated types stay ergonomic.
function stripTupleConstraints(node) {
  if (node && typeof node === 'object') {
    if (node.type === 'array') {
      delete node.minItems;
      delete node.maxItems;
    }
    for (const value of Object.values(node)) {
      stripTupleConstraints(value);
    }
  }
}

async function generate(schemaPath, outPath, interfaceName) {
  const raw = await fs.readFile(schemaPath, 'utf-8');
  const schema = JSON.parse(raw);
  const generatorSchema = JSON.parse(JSON.stringify(schema));
  stripTupleConstraints(generatorSchema);

  const banner = `/* eslint-disable */
/**
 * Automatically generated from ${path.relative(process.cwd(), schemaPath)}. DO NOT EDIT MANUALLY.
 */`;
  const ts = await compile(generatorSchema, interfaceName, {
    bannerComment: banner,
    unreachableDefinitions: true,
    strictIndexSignatures: false,
  });

  await fs.mkdir(path.dirname(outPath), { recursive: true });
  await fs.writeFile(outPath, ts, 'utf-8');
  console.log(`Generated ${outPath} from ${schemaPath}`);
}

async function main() {
  await generate(
    path.resolve('packages/visual-tools/src/visual.schema.json'),
    path.resolve('packages/visual-tools/src/visual.generated.ts'),
    'VisualContracts'
  );
  await generate(
    path.resolve('web/src/generated/api.schema.json'),
    path.resolve('web/src/generated/api.generated.ts'),
    'GuidanceApiContracts'
  );
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
