import { describe, expect, it } from "vitest";
import { normalizeLatex } from "./latex";

describe("normalizeLatex", () => {
  it("leaves correctly escaped commands unchanged", () => {
    expect(normalizeLatex("H(S)=-\\sum_c p(c)\\log_2 p(c)")).toBe(
      "H(S)=-\\sum_c p(c)\\log_2 p(c)",
    );
  });

  it("collapses doubled command slashes from serialized formulas", () => {
    expect(normalizeLatex("H(S)=-\\\\sum_c p(c)\\\\log_2 p(c)")).toBe(
      "H(S)=-\\sum_c p(c)\\log_2 p(c)",
    );
  });
});
