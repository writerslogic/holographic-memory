import assert from "node:assert/strict";
import test from "node:test";
import vm from "node:vm";
import { destination, redirectHtml } from "./build-docs-redirects.mjs";

test("legacy documentation redirects preserve route, query and fragment", () => {
  for (const version of ["v0.6.4", "v0.6.3", "v0.6.1", "latest"]) {
    let result;
    const pathname = `/${version}/concepts/`;
    const html = redirectHtml(pathname);
    const script = html.match(/<script>([\s\S]*?)<\/script>/i)[1];
    vm.runInNewContext(script, {
      location: { pathname, search: "?a=1&b=2", hash: "#retention", replace: (url) => { result = url; } },
    });
    assert.equal(result, "https://writerslogic.com/warm/docs/latest/concepts/?a=1&b=2#retention");
    assert.equal(destination(pathname), "https://writerslogic.com/warm/docs/latest/concepts/");
  }
});

test("route text cannot create markup or enter the redirect script", () => {
  const html = redirectHtml('/latest/"/><script>throw 123</script>&');
  assert.equal((html.match(/<script>/gi) || []).length, 1);
  assert.ok(!html.includes("<script>throw 123"));
  assert.ok(html.includes("&quot;"));
  assert.ok(html.includes("&lt;script>"));
  const script = html.match(/<script>([\s\S]*?)<\/script>/i)[1];
  assert.ok(!script.includes("throw 123"));
  assert.ok(!redirectHtml("/", true).includes('http-equiv="refresh"'));
});
