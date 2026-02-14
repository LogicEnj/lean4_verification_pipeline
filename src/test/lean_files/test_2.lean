
import Mathlib

theorem t (n : ℕ) (h : n > 30 ∧ n < 50) (h1 : ¬∃ k, 0 < k ∧ n % k = 0) (h2 : ¬∃ k, 0 < k ∧ (n + 2) % k = 0) : n = 41 := by
  rcases h with ⟨h1l, h1u⟩
  interval_cases n <;> norm_num at *
  <;> tauto
