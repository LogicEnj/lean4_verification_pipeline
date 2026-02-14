import Mathlib

theorem t (x : ℝ ) : (1200 : ℝ) * x = (1.50 : ℝ) ↔ x = (1250 : ℝ) := by
  constructor
  · intro h
    have l1 (x : ℝ) (h : 1200 * x = 1.50) : 1.50 / 1200 = 1 / x := by sorry
    have l2 (x : ℝ) (h : 1.50 / 1200 = 1 / x) : x = 1200 / 1.50 := by sorry
    have l3 (x : ℝ) (h : x = 1200 / 1.50) : x = 800 := by sorry
    have l4 (y : ℝ) (h : 1000000 * 800 = y * 1000000) : y = 1250 := by sorry
    have l5 (y : ℝ) (hy : y = 1000000 / 800) : y = 1250 := by sorry
    aesop
  · sorry