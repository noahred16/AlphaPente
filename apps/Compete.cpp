#include "MCTS.hpp"
#include "PenteGame.hpp"
#include <iostream>
#include <cstring>


// How to run: ./compete "1	K10	K9 2	K6	L11 3	M8	J11 4	L7	N9 5	J5	H4"
int main(int argc, char* argv[]) {
    std::cout << "Playing Pente..." << std::endl;

    // override argv for testing
    // const char* hardCodedGame = "1. K10 L9 2. G10 L7 3. M10 L8 4. L10 J10 5. J12 L6 6. L5 K9 7. H11 K13 8. K11 K12 9. K11 M9 10. F9 E8 11. K14 K13 12. H13 G14 13. N9 M7";
    const char* hardCodedGame = "1. K10 L9 2. G10 L7 3. M10 L8 4. L10 J10 5. J12 L6 6. L5 K9 7. H11 K13 8. K11 K12 9. K11 M9 10. F9 E8 11. K14 K13 12. H13 G14 13. N9 M7 14. N6 K7 15. N10";

    // use argv[1] if provided, else use hardcoded
    const char* gameDataStr = (argc >= 2) ? argv[1] : hardCodedGame;

    // Print the received game data string
    std::cout << "Game Data String: " << gameDataStr << std::endl;

    // Parse the game data string
    std::vector<std::string> moves;
    char* gameDataCopy = strdup(gameDataStr); // Duplicate to avoid modifying argv
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
    config.maxIterations = 150000; //150,000
    // config.maxIterations = 100000; //100,000
    config.explorationConstant = 1.414;

    MCTS mcts(config);
    // PenteGame::Move _ = mcts.search(game);
    mcts.search(game);
    mcts.printStats();
    mcts.printBestMoves(15);
    // print see branch
    // std::cout << "Printing branch for move G11:\n";
    // mcts.printBranch("N6", 10);
    // mcts.printBranch("G11", 10);

    return 0;
}


