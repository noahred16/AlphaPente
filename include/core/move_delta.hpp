#pragma once

namespace core {

struct MoveDelta {
    int dx;
    int dy;
    
    MoveDelta(int dx, int dy) : dx(dx), dy(dy) {}
};

} // namespace core