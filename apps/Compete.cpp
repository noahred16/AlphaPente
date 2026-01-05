#include "MCTS.hpp"
#include "PenteGame.hpp"
#include <iostream>
#include <cstring>


// How to run: ./compete "1	K10	K9 2	K6	L11 3	M8	J11 4	L7	N9 5	J5	H4"
int main(int argc, char* argv[]) {
    std::cout << "Playing Pente..." << std::endl;

    // arguments:
    // game data string (required)
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <game_data_string>" << std::endl;
        return 1;
    }

    const char* gameDataStr = argv[1];

    // Print the received game data string
    std::cout << "Game Data String: " << gameDataStr << std::endl;

    // Parse the game data string
    std::vector<std::string> moves;
    char* gameDataCopy = strdup(argv[1]); // Duplicate to avoid modifying argv
    char* token = std::strtok(gameDataCopy, " \t");
    
    while (token != nullptr) {
        moves.push_back(std::string(token));
        token = std::strtok(nullptr, " \t");
    }
    free(gameDataCopy);

    // debug move array
    std::cout << "Parsed moves:\n";
    for (const auto& moveStr : moves) {
        std::cout << moveStr << " ";
    }
    std::cout << std::endl;

    // Game time
    PenteGame game;
    game.reset();

    // Replay the moves on the PenteGame instance
    for (size_t i = 0; i < moves.size(); i++) {
        // Skip move number (every 3rd token)
        if (i % 3 == 0) {
            continue;
        }
        const std::string& moveStr = moves[i];
        const char* moveCStr = moveStr.c_str();
        game.makeMove(moveCStr);
    }
        
    game.print();

    // MCTS configuration
    MCTS::Config config;
    config.maxIterations = 100000;
    config.explorationConstant = 1.414;

    MCTS mcts(config);
    PenteGame::Move bestMove = mcts.search(game);
    mcts.printStats();
    mcts.printBestMoves(10);
    // mcts.printBranch("idk", 10);

    return 0;
    

    // TODO: Implement compete logic

    return 0;
}


