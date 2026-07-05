#pragma once
#include "BitBoard.hpp"

// Renju forbidden-move rules for Black. Every check here is hypothetical: (x, y) must be
// empty in `black`, and the function evaluates the position as if Black had just played there.
namespace RenjuRules {

// True if placing a Black stone at (x, y) creates a run of 6 or more stones in some
// direction. Six-or-more is "overline" for Black: it does not win, unlike an exact five.
bool isOverline(const BitBoard &black, int x, int y);

} // namespace RenjuRules
