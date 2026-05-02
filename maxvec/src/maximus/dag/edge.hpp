#pragma once

class Edge {
public:
    int source_port = -1;
    int target_port = -1;

    Edge() = default;

    Edge(int source_port, int target_port): source_port(source_port), target_port(target_port) {}
};