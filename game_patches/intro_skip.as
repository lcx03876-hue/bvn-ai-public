// ==================== INJECTED: skip the round intro (not in the original) ====================
// Why: bring the match up as fast as possible. The stock intro (camera moves,
// voice lines, poses) runs 40-60 seconds, during which actionEnable is false
// while the CPU opponent is already hitting P1 -- measured at 240 HP left by
// the time control was handed over.
// `skip()` already exists in the original code (the pause menu uses it), so
// this simply calls it at the top of renderStart1v1().
// Measured effect: intro drops to about 15 frames (_holdFrame = 0.5 * FPS).
skip();
// ================== end of intro-skip injection ==================
