-- EvenPlusOdd.lean
import Mathlib.Tactic

-- Define what it means for a number to be even
def even (n : ℕ) : Prop := ∃ k, n = 2 * k

-- Define what it means for a number to be odd
def odd (n : ℕ) : Prop := ∃ k, n = 2 * k + 1

-- The theorem statement
theorem even_plus_odd : ∀ n m, even n → odd m → odd (n + m) := by
  -- The proof
  intro n m hn hm
  cases' hn with k hk
  cases' hm with l hl
  us k + l
  rw [hk, hl]
  ring
