/**
 * Framework for NoGo and similar games (C++ 11)
 * agent.h: Define the behavior of variants of the player
 *
 * Author: Theory of Computer Games
 *         Computer Games and Intelligence (CGI) Lab, NYCU, Taiwan
 *         https://cgilab.nctu.edu.tw/
 */

#pragma once
#include <string>
#include <random>
#include <sstream>
#include <map>
#include <type_traits>
#include <algorithm>
#include <fstream>
#include <thread>
#include <future>
#include <ctime>
#include <unordered_map>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <cmath>
#include "board.h"
#include "action.h"

using namespace std;

class MCTS {
public:
    struct Node {
        int Tn;
        int x;
        int raveTn;
        int rave_x;
        board state; // current board state       
        board::point parent_move;
        vector<Node*> child2board; // keep action according to the board position
        vector<board::point> legal;
        vector<Node*> children;

        Node(board b, std::default_random_engine& engine) : Tn(0), x(0), raveTn(0), rave_x(0), state(b) {
            child2board.resize(board::size_x * board::size_y, NULL);
            for (int i = 0; i < board::size_x * board::size_y; i++) {
                board::point move(i);
                board tmp = b;
                if (tmp.place(move) == board::legal)
                    legal.push_back(move);
            }
            std::shuffle(legal.begin(), legal.end(), engine);
        }
    };

    MCTS(){
        engine.seed(1234);
        boardSize = board::size_x * board::size_y;
        visited.resize(boardSize, false);
        actions.reserve(boardSize);
        for (int i = 0; i < boardSize; i++)
            actions.push_back(board::point(i));
    }

    void run(const board& b, int simulation_time, float constant) {
        nodePool.reserve(simulation_time + 2);
        root = new Node(b, engine);
        nodePool.push_back(*root);
        c = constant;
        int time = 0;

        while(1){
            traverse(root);
            if(simulation_time == (++time))
                break;
        }
    }

    int getTn(int point) {
        if (root->child2board[point] != NULL)
            return root->child2board[point]->Tn;
        else
            return 0;
    }

    int traverse(Node* node, bool isOpponent=false) {
        if (!node->legal.empty()) {  // expand and simulate
            Node* leaf = expansion(node);
            int result = simulation(leaf->state, !isOpponent);
            backpropagation(leaf, result);
            backpropagation(node, result);
            return result;
        } 
        else {
            int result;
            if (node->children.empty()) {  // Terminal node
                result = simulation(node->state, isOpponent);
            } 
            else {
                Node* nextNode = selection(node, isOpponent);
                result = traverse(nextNode, !isOpponent);
                visited[nextNode->parent_move.i] = true;
            }
            backpropagation(node, result);
            return result;
        }
    }

    Node* selection(Node* node, bool isOpponent) {
        float maxUCT = -1;
        Node* bestChild = nullptr;
        for(size_t i=0;i<node->children.size();i++){
            if (node->children[i]->Tn == 0){
                // not explore
                bestChild = node->children[i];
                return bestChild;
            }
            double val;
            calculate_UCT(*(node->children[i]), node->Tn, isOpponent, val);
            if (maxUCT < val) {
                maxUCT = val;
                bestChild = node->children[i];
            } 
        }

        if (maxUCT == -1) 
            exit(0);
        
        return bestChild;
    }

    int simulation(const board& state, bool isOpponent) {
        std::vector<board::point> empty;
        for (int i = 0; i < boardSize; i++) {
            board::point move(i);
            if (state[move.x][move.y] == board::empty)
                empty.push_back(move);
        }
        board after = state;
        int n = empty.size();
        if(n == 0)
            exit(0);
        
        while(1){
            int i = 0;
            board tmp = after;
            board::point nextMove;
        
            while(i < n){
                std::uniform_int_distribution<int> uniform(i, n - 1);
                int index = uniform(engine);
                if (tmp.place(empty[index]) == board::legal) {
                    std::swap(empty[index], empty[n-1]);
                    nextMove = empty[n-1];
                    break;
                } 
                else {
                    std::swap(empty[index], empty[i]);
                    i++;
                }
                if(i == n - 1)
                    nextMove = empty[0];
            }

            if(after.place(nextMove) != board::legal)
                return isOpponent;

            isOpponent = !isOpponent;
            n--;
        }
    }

    Node* expansion(Node* node) {
        board after = node->state;
        board::point move = node->legal.back();
        node->legal.pop_back();
        after.place(move);
        
        nodePool.push_back(Node(after, engine));
        nodePool.back().parent_move = move;
        node->children.push_back(&nodePool.back());
        node->child2board[move.i] = &nodePool.back();
        
        return node->children.back();
    }

    void backpropagation(Node* node, int result) {
        node->Tn++;
        node->x += result;
        
        for(Node* child : node->children){
            child->raveTn++;
            child->rave_x += result;
        }
    }

    void calculate_UCT(Node& node, int N, bool isOpponent, double& UCT_val) {
        if (node.Tn == 0) return;
        double beta = (double)node.raveTn / ((double)node.Tn + (double)node.raveTn + 4 * (double)node.Tn * (double)node.raveTn * 0.025 * 0.025);
        double winRate = (double)node.x / (double)(node.Tn + 1);
        double raveWinRate = (double)node.rave_x / (double)(node.raveTn + 1);
        double exploit = (isOpponent) ? (1 - beta) * (1 - winRate) + beta * (1 - raveWinRate) : (1 - beta) * winRate + beta * raveWinRate;
        double explore = sqrt(log(N) / (double)(node.Tn + 1));
        UCT_val = exploit + c * explore;
    }

private:
    Node* root;
    float c;
    vector<board::point> actions;
    vector<Node> nodePool;
    std::vector<bool> visited;
    board::piece_type who;
    default_random_engine engine;
    uniform_int_distribution<int> uniform;
    int boardSize;
};

class agent {
public:
	agent(const std::string& args = "") {
		std::stringstream ss("name=unknown role=unknown " + args);
		for (std::string pair; ss >> pair; ) {
			std::string key = pair.substr(0, pair.find('='));
			std::string value = pair.substr(pair.find('=') + 1);
			meta[key] = { value };
		}
	}
	virtual ~agent() {}
	virtual void open_episode(const std::string& flag = "") {}
	virtual void close_episode(const std::string& flag = "") {}
	virtual action take_action(const board& b) { return action(); }
	virtual bool check_for_win(const board& b) { return false; }

public:
	virtual std::string property(const std::string& key) const { return meta.at(key); }
	virtual void notify(const std::string& msg) { meta[msg.substr(0, msg.find('='))] = { msg.substr(msg.find('=') + 1) }; }
	virtual std::string name() const { return property("name"); }
	virtual std::string role() const { return property("role"); }

protected:
	typedef std::string key;
	struct value {
		std::string value;
		operator std::string() const { return value; }
		template<typename numeric, typename = typename std::enable_if<std::is_arithmetic<numeric>::value, numeric>::type>
		operator numeric() const { return numeric(std::stod(value)); }
	};
	std::map<key, value> meta;
};

/**
 * base agent for agents with randomness
 */
class random_agent : public agent {
public:
	random_agent(const std::string& args = "") : agent(args) {
		if (meta.find("seed") != meta.end())
			engine.seed(int(meta["seed"]));
		if (meta.find("simulation") != meta.end())
			simulation_time = (int(meta["simulation"]));
		if (meta.find("parallel") != meta.end())
			numOfThread = (int(meta["parallel"]));
	}
	virtual ~random_agent() {}

protected:
	std::default_random_engine engine;
	int simulation_time;
	int numOfThread;
};

/**
 * random player for both side
 * put a legal piece randomly
 */
class MCTS_player : public random_agent {
public:
	MCTS_player(const std::string& args = "") : random_agent("name=random role=unknown " + args),
		space(board::size_x * board::size_y), who(board::empty) {
		if (name().find_first_of("[]():; ") != std::string::npos)
			throw std::invalid_argument("invalid name: " + name());
		if (role() == "black") who = board::black;
		if (role() == "white") who = board::white;
		if (who == board::empty)
			throw std::invalid_argument("invalid role: " + role());
		for (size_t i = 0; i < space.size(); i++)
			space[i] = action::place(i, who);
	}

	virtual action take_action(const board& state) {
        vector<MCTS> mcts(numOfThread);
        vector<thread> threads;
        for (int i = 0; i < numOfThread; i++) {
            threads.push_back(thread(&MCTS_player::runMCTS, this, state, &mcts[i]));
        }
        for (int i = 0; i < numOfThread; i++) {
            threads[i].join();
        }

        int max_count = -1;
        int key = 0;
        for (int i = 0; i < int(space.size()); i++) {
            int total = 0;
            for (int j = 0; j < numOfThread; j++)
                total += mcts[j].getTn(i);
            if (max_count < total) {
                max_count = total;
                key = i;
            }
        }

        return action::place(board::point(key), who);
	}

    void runMCTS(board state, MCTS* mcts) {
        mcts->run(state, simulation_time, 0.5);
    }

private:
	std::vector<action::place> space;
	board::piece_type who;
};