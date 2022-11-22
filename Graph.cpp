// Graph.cpp : This file contains the 'main' function. Program execution begins and ends there.
//


#include <iostream>
#include <vector>
#include <list>
#include <queue>
#include <set>
#include <ctime>
#include <cstdlib>
#include <bitset>
#include <ratio>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <string>
#include <sstream>
#include <stack>
#include <fstream>

struct Point {
	double x;
	double y;
	Point(double _x, double _y) {
		x = _x;
		y = _y;
	}
};

double distance(Point a, Point b) {
	return pow((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y), 0.5);
}

struct Iris {
	double sepalLength;
	double sepalWidth;
	double petalLength;
	double petalWidth;
};

double irisDistance(Iris a, Iris b) {
	return pow(pow(a.sepalLength - b.sepalLength, 2) + pow(a.sepalWidth - b.sepalWidth, 2) +
		pow(a.petalLength - b.petalLength, 2) + pow(a.petalWidth - b.petalWidth, 2), 0.5);
}

std::vector<Iris> readIrisesFromFile(std::string fileName) {
	std::ifstream file;
	std::vector<Iris> irises;
	file.open(fileName);
	int i = 0;
	std::string line;
	while (std::getline(file, line)) {
		if (line == "") break;
		std::istringstream iss(line);
		std::string substr;
		Iris iris;
		std::getline(iss, substr, ',');
		iris.sepalLength = std::stod(substr);
		std::getline(iss, substr, ',');
		iris.sepalWidth = std::stod(substr);
		std::getline(iss, substr, ',');
		iris.petalLength = std::stod(substr);
		std::getline(iss, substr, ',');
		iris.petalWidth = std::stod(substr);
		//iss >> iris.sepalLength >> iris.sepalWidth >> iris.petalLength >> iris.petalWidth;
		irises.push_back(iris);
	}
	file.close();
	return irises;
}

template<class T> class Node {
public:
	T data;
	Node* next;
	Node(T val, Node* nxt = nullptr) {
		data = val;
		next = nxt;
	}
};
/*
template<class T> class List {
private:
	Node<T>* first;
public:
	List() {
		first = nullptr;
	}
	List(const List& lst) : List() {
		Node<T>* current = nullptr;
		for (iterator it = lst.begin(); it != end(); ++it) {
			current = insert_after(it->data, current);
		}
	}
	~List() {
		iterator it = begin();
		while (it != end()) {
			iterator nit(it->next);
			delete* it;
			it = nit;
		}
	}
	List<T>& operator=(const List<T> lst) {
		first = nullptr;
		Node<T>* current = nullptr;
		for (iterator it = lst.begin(); it != end(); ++it) {
			current = insert_after(it->data, current);
		}
		return *this;
	}
	Node<T>* getFirst() {
		return first;
	}
	T front() {
		return first->data;
	}
	bool empty() {
		return first == nullptr;
	}
	void clear() {
		iterator it = begin();
		while (it != end()) {
			iterator nit(it->next);
			delete* it;
			it = nit;
		}
		first = nullptr;
	}
	Node<T>* insert_after(T data, Node<T>* n) {
		if (n == nullptr) return push_front(data);
		Node<T>* pnn = new Node<T>(data, n->next);
		n->next = pnn;
		return pnn;
	}
	void erase_after(Node<T>* n) {
		if (n == nullptr) {
			pop_front();
			return;
		}
		Node<T>* rem = n->next;
		n->next = rem->next;
		delete rem;
	}
	Node<T>* push_front(T data) {
		Node<T>* pnn = new Node<T>(data, first);
		first = pnn;
		return pnn;
	}
	void pop_front() {
		Node<T>* rem = first;
		first = rem->next;
		delete rem;
	}
	class iterator {
	private:
		Node<T>* current;
	public:
		iterator() {
			current = first;
		}
		iterator(Node<T>* pnode) {
			current = pnode;
		}
		iterator(const iterator& it) {
			current = it.current;
		}
		Node<T>*& operator*() {
			return current;
		}
		Node<T>* operator->() {
			return current;
		}
		iterator operator++() {
			current = current->next;
			return current;
		}
		bool operator==(iterator it) {
			return it.current == current;
		}
		bool operator!=(iterator it) {
			return !(it == *this);
		}
	};
	iterator begin() const {
		iterator it(first);
		return it;
	}
	iterator end() const {
		iterator it(nullptr);
		return it;
	}
};
*/
template<class T> struct Pair {
	unsigned int first;
	T second;
	Pair() {

	}
	Pair(unsigned int n1, T n2) {
		first = n1;
		second = n2;
	}
	Pair(const Pair<T>& p) {
		first = p.first;
		second = p.second;
	}
	Pair& operator=(const Pair p) {
		first = p.first;
		second = p.second;
		return *this;
	}
};


template<class T> struct Vertex {
	Pair<T> data;
	std::vector<Vertex<T>*> adjacencyList;
	std::vector<int> weights;
	Vertex() {

	}
	Vertex(Pair<T> d) {
		data = d;
	}
	Vertex(Pair<T> d, std::vector<Vertex<T>*> al, std::vector<int> ws) {
		data = d;
		adjacencyList = al;
		weights = ws;
	}
	Vertex(const Vertex<T>& vx) {
		data = vx.data;
		adjacencyList = vx.adjacencyList;
		weights = vx.weights;
	}
	Vertex<T>& operator=(const Vertex<T> vx) {
		data = vx.data;
		adjacencyList = vx.adjacencyList;
		weights = vx.weights;
		return *this;
	}
};

struct EdgeInfo {
	int a;
	int b;
	int weight;
	bool operator<(const EdgeInfo ei) {
		if (weight < ei.weight)
			return true;
		else return false;
	}
};

template <class T> struct VertexDistance {
	Vertex<T>* vertex;
	int distance;
	VertexDistance() {

	}
	VertexDistance(Vertex<T>* pvx, int d) {
		vertex = pvx;
		distance = d;
	}
	bool operator<(const VertexDistance<T> vd) const {
		if (distance < vd.distance || (distance == vd.distance && vertex->data.first < vd.vertex->data.first))
			return true;
		else return false;
	}
	VertexDistance(const VertexDistance& vd) {
		vertex = vd.vertex;
		distance = vd.distance;
	}
};

const int UNREACHABLE = -1;

template<class T> class Graph {
protected:
	void convertQueueToBitset(std::queue<Vertex<T>*> q, std::vector<bool>& frontier) {
		std::fill(frontier.begin(), frontier.end(), false);
		while (!q.empty()) {
			frontier[index(q.front())] = true;
			q.pop();
		}
	}
	void convertBitsetToQueue(std::vector<bool> frontier, std::queue<Vertex<T>*>& q) {
		while (!q.empty()) {
			q.pop();
		}
		for (int i = 0; i < frontier.size(); i++)
			if (frontier[i])
				q.push(&(vertices[i]));
	}
	void BFS(Vertex<T>* current, std::vector<bool>& markedVertices,
		std::vector<Vertex<T>*>& path, std::vector<int>& distances, std::vector<Vertex<T>*>& predecessors, std::queue<Vertex<T>*>& q,
		double a) {
		bool iterationComplete = false;
		while (true) {
			Vertex<T>* u = q.front();
			q.pop();
			for (int i = 0; i < u->adjacencyList.size(); i++) {
				Vertex<T>* it = u->adjacencyList[i];
				if (!(markedVertices[index(it)])) {
					markedVertices[index(it)] = true;
					path.push_back(it);
					distances[index(it)] = distances[index(u)] + 1;
					predecessors[index(it)] = u;
					q.push(it);
				}
			}
			if (!q.empty() && distances[index(u)] + 1 == distances[index(q.front())]) {
				std::queue<Vertex<T>*> q2 = q;
				int mf = 0, mu = 0;
				while (!q2.empty()) {
					Vertex<T>* vx = q2.front();
					q2.pop();
					for (int i = 0; i < vx->adjacencyList.size(); i++) {
						Vertex<T>* it = vx->adjacencyList[i];
						if (distances[index(it)] == UNREACHABLE || distances[index(it)] == distances[index(u)] + 1)
							mf++;
					}
					for (int i = 0; i < vertices.size(); i++)
						if (predecessors[i] == nullptr) {
							for (int j = 0; j < vertices[i].adjacencyList.size(); j++) {
								Vertex<T>* it = vertices[i].adjacencyList[j];
								mu++;
							}
						}
				}
				if (mf > mu / a && iterationComplete) return;
				else iterationComplete = true;
			}
			if (q.empty()) return;
		}
	}
	void BFSBottomUp(Vertex<T>* current, std::vector<bool>& markedVertices,
		std::vector<Vertex<T>*>& path, std::vector<int>& distances, std::vector<Vertex<T>*>& predecessors, std::vector<bool>& frontier,
		double b) {
		std::vector<bool> next(vertices.size());
		bool iterationComplete = false;
		predecessors[index(current)] = current;
		distances[index(current)] = 0;
		while (std::any_of(frontier.begin(), frontier.end(), [](bool v) {return v; })) {
			for (int i = 0; i < vertices.size(); i++)
				if (predecessors[i] == nullptr) {
					for (int j = 0; j < vertices[i]->adjacencyList.size(); j++) {
						Vertex<T>* it = vertices[i]->adjacencyList[j];
						if (frontier[index(it)]) {
							next[i] = true;
							predecessors[i] = it;
							distances[i] = distances[index(it)] + 1;
							path.push_back(&(vertices[i]));
							break;
						}
					}
				}
			frontier = next;
			std::fill(next.begin(), next.end(), false);
			int nf = std::count(frontier.begin(), frontier.end(), true);
			if (nf < vertices.size() && iterationComplete) return;
			else iterationComplete = true;
		}
	}
	void collectPaths(int source, int dest, std::vector<std::vector<Vertex<T>*>>& paths, std::vector<std::vector<Vertex<T>*>> predecessors) {
		/*if (predecessors[dest].size() == 0) {
			predecessors.push_back({});
			return;
		}*/
		if (predecessors[dest].size() == 0 || index(predecessors[dest][0]) == source) {
			paths.push_back({ &(vertices[dest]) });
			return;
		}
		int depth;
		std::stack<int> st, depths;
		int rememberedDepth = 0;
		std::vector<Vertex<T>*> currPath = {  };
		st.push(dest);
		depths.push(1);
		while (!st.empty()) {
			int node = st.top();


			//if (node == source)
				//return;
			st.pop();
			depth = depths.top();
			depths.pop();
			for (int i = 0; i < rememberedDepth - depth + 1; i++)
				currPath.pop_back();
			if (node != source)
				currPath.push_back(&(vertices[node]));
			for (int i = 0; i < predecessors[node].size(); i++) {
				if (index(predecessors[node][i]) == source) {
					paths.push_back(currPath);
					continue;
				}

				//paths[paths.size() - 1].push_back(predecessors[node][i]);
				st.push(index(predecessors[node][i]));
				depths.push(depth + 1);
				//collectPaths(source, index(predecessors[dest][i]), paths, predecessors, depth + 1);
				//if(newPath) currPaths.pop_back();
			}
			rememberedDepth = depth;
		}
	}
	void brandesBFS(int startingVertex, std::vector<int> clusters, std::vector<std::vector<Vertex<T>*>>& predecessors,
		std::vector<double>& deltas, std::vector<int>& sigmas, std::vector<double>& cbs/*, std::vector<double>* localDeltas = nullptr*/) {
		//std::vector<bool> markedVertices(vertices.size());
		//std::vector<std::vector<Vertex<T>*>> predecessors(vertices.size());
		int currCluster = clusters[startingVertex];
		std::vector<std::vector<std::vector<Vertex<T>*>>> paths(vertices.size());
		//markedVertices[startingVertex] = true;
		std::vector<Vertex<T>*> path = {};
		path.push_back(&(vertices[startingVertex]));
		std::vector<int> distances(vertices.size());
		for (int i = 0; i < vertices.size(); i++) {
			if (i == startingVertex) distances[i] = 0;
			else distances[i] = UNREACHABLE;
		}
		std::queue<Vertex<T>*> q;
		std::stack<Vertex<T>*> s;
		q.push(&(vertices[startingVertex]));
		while (!q.empty()) { //
			Vertex<T>* u = q.front();
			q.pop();
			s.push(u);
			for (int i = 0; i < u->adjacencyList.size(); i++) {
				Vertex<T>* it = u->adjacencyList[i];
				if (clusters[index(it)] != currCluster) continue;
				if (distances[index(it)] == UNREACHABLE) {
					//markedVertices[index(it)] = true;
					//path.push_back(it);
					distances[index(it)] = distances[index(u)] + 1;
					//predecessors[index(it)] = u;
					q.push(it);
				}
				if (distances[index(u)] + 1 == distances[index(it)]) {
					predecessors[index(it)].push_back(u);
					sigmas[index(it)] += sigmas[index(u)];
				}
			}
		} //
		while (!s.empty()) {
			int w = index(s.top());
			s.pop();
			for (int i = 0; i < predecessors[w].size(); i++) {
				deltas[index(predecessors[w][i])] += double(sigmas[index(predecessors[w][i])]) / sigmas[w] * (1 + deltas[w]);
			}
			if (w != startingVertex) cbs[w] += deltas[w];
		}
	}
	void modifiedBrandesV1BFS(int startingVertex, std::vector<int> clusters, std::vector<std::vector<Vertex<T>*>>& predecessors,
		std::vector<double>& deltas, std::vector<int>& sigmas, std::vector<double>& localDeltas, std::vector<int>& localSigmas, std::vector<int>& distances,
		std::vector<bool> externalNodes) {
		//std::vector<bool> markedVertices(vertices.size());
		//std::vector<std::vector<Vertex<T>*>> predecessors(vertices.size());
		int currCluster = clusters[startingVertex];
		//std::vector<std::vector<std::vector<Vertex<T>*>>> paths(vertices.size());
		//markedVertices[startingVertex] = true;
		//std::vector<Vertex<T>*> path = {};
		//path.push_back(&(vertices[startingVertex]));
		for (int i = 0; i < vertices.size(); i++) {
			if (i == startingVertex) distances[i] = 0;
			else distances[i] = UNREACHABLE;
		}
		std::queue<Vertex<T>*> q;
		std::stack<Vertex<T>*> s;
		q.push(&(vertices[startingVertex]));
		while (!q.empty()) { //
			Vertex<T>* u = q.front();
			q.pop();
			s.push(u);
			for (int i = 0; i < u->adjacencyList.size(); i++) {
				Vertex<T>* it = u->adjacencyList[i];
				if (clusters[index(it)] != currCluster) continue;
				if (distances[index(it)] == UNREACHABLE) {
					//markedVertices[index(it)] = true;
					//path.push_back(it);
					distances[index(it)] = distances[index(u)] + 1;
					//predecessors[index(it)] = u;
					q.push(it);
				}
				if (distances[index(u)] + 1 == distances[index(it)]) {
					predecessors[index(it)].push_back(u);
					sigmas[index(it)] += sigmas[index(u)];
				}
			}
		} //
		while (!s.empty()) {
			int w = index(s.top());
			s.pop();
			for (int i = 0; i < predecessors[w].size(); i++) {
				deltas[index(predecessors[w][i])] += double(sigmas[index(predecessors[w][i])]) / sigmas[w] * (int(!externalNodes[w]) + deltas[w]);
			}
			if (w != startingVertex) {
				if (externalNodes[w]) continue;
				localDeltas[w] += deltas[w];
			}
			localSigmas[w] += sigmas[w];
		}
	}
	void modifiedBrandesV2BFS(int startingVertex, std::vector<int> clusters, std::vector<std::vector<Vertex<T>*>>& predecessors,
		std::vector<double>& globalDeltas, int nClusters) {
		//std::vector<bool> markedVertices(vertices.size());
		//std::vector<std::vector<Vertex<T>*>> predecessors(vertices.size());
		int currCluster = clusters[startingVertex];
		std::vector<std::vector<double>> deltas(vertices.size());
		for (int i = 0; i < vertices.size(); i++) {
			deltas[i].resize(nClusters);
		}
		std::vector<int> sigmas(vertices.size());
		sigmas[startingVertex] = 1;
		for (int i = 0; i < nClusters; i++) {
			deltas[startingVertex][i] = 1;
		}
		std::vector<int> distances(vertices.size());
		//std::vector<std::vector<std::vector<Vertex<T>*>>> paths(vertices.size());
		//markedVertices[startingVertex] = true;
		//std::vector<Vertex<T>*> path = {};
		//path.push_back(&(vertices[startingVertex]));
		for (int i = 0; i < vertices.size(); i++) {
			if (i == startingVertex) distances[i] = 0;
			else distances[i] = UNREACHABLE;
		}
		std::queue<Vertex<T>*> q;
		std::stack<Vertex<T>*> s;
		q.push(&(vertices[startingVertex]));
		while (!q.empty()) { //
			Vertex<T>* u = q.front();
			q.pop();
			s.push(u);
			for (int i = 0; i < u->adjacencyList.size(); i++) {
				Vertex<T>* it = u->adjacencyList[i];
				//if (clusters[index(it)] != currCluster && clusters[index(u)] != currCluster) continue;
				if (distances[index(it)] == UNREACHABLE) {
					//markedVertices[index(it)] = true;
					//path.push_back(it);
					distances[index(it)] = distances[index(u)] + 1;
					//predecessors[index(it)] = u;
					q.push(it);
				}
				if (distances[index(u)] + 1 == distances[index(it)]) {
					predecessors[index(it)].push_back(u);
					sigmas[index(it)] += sigmas[index(u)];
				}
			}
		} //
		while (!s.empty()) {
			int w = index(s.top());
			s.pop();
			for (int i = 0; i < predecessors[w].size(); i++) {
				for (int j = 0; j < nClusters; j++)
					deltas[index(predecessors[w][i])][j] +=
					double(sigmas[index(predecessors[w][i])]) / sigmas[w] * (int(clusters[w] == j) + deltas[w][j]);
			}
			if (w != startingVertex) {
				if (clusters[w] != currCluster) {
					globalDeltas[w] += 2 * deltas[w][clusters[w]];
					//globalDeltas[w] += deltas[w][clusters[w]];
					for (int j = 0; j < nClusters; j++) {
						if (j == clusters[w]) continue;
						else globalDeltas[w] += deltas[w][j];
					}
				}
				//localSigmas[w] += sigmas[w];
			}
		}
	}
	std::vector<bool> findBorderNodes(std::vector<int> clusters) {
		std::vector<bool> res(vertices.size());
		for (int i = 0; i < vertices.size(); i++) {
			if (res[i]) continue;
			int currCluster = clusters[i];
			for (int j = 0; j < vertices[i].adjacencyList.size(); j++) {
				Vertex<T>* it = vertices[i].adjacencyList[j];
				if (clusters[index(it)] != currCluster) {
					res[i] = true;
					break;
				}
			}
		}
		return res;
	}
	void localBFSAllPaths(int startingVertex, std::vector<int> clusters, std::vector<std::vector<Vertex<T>*>>& predecessors) {
		//std::vector<bool> markedVertices(vertices.size());
		//std::vector<std::vector<Vertex<T>*>> predecessors(vertices.size());
		int currCluster = clusters[startingVertex];
		//std::vector<std::vector<std::vector<Vertex<T>*>>> paths(vertices.size());
		//markedVertices[startingVertex] = true;
		std::vector<Vertex<T>*> path = {};
		path.push_back(&(vertices[startingVertex]));
		std::vector<int> distances(vertices.size());
		for (int i = 0; i < vertices.size(); i++) {
			if (i == startingVertex) distances[i] = 0;
			else distances[i] = UNREACHABLE;
		}
		std::queue<Vertex<T>*> q;
		q.push(&(vertices[startingVertex]));
		while (!q.empty()) { //
			Vertex<T>* u = q.front();
			q.pop();
			for (int i = 0; i < u->adjacencyList.size(); i++) {
				Vertex<T>* it = u->adjacencyList[i];
				if (clusters[index(it)] != currCluster) continue;
				if (distances[index(it)] == UNREACHABLE) {
					//markedVertices[index(it)] = true;
					//path.push_back(it);
					distances[index(it)] = distances[index(u)] + 1;
					//predecessors[index(it)] = u;
					q.push(it);
				}
				if (distances[index(u)] + 1 == distances[index(it)]) {
					predecessors[index(it)].push_back(u);
				}
			}
		} //
	}
	std::vector<bool> buildHSN(std::vector<int> clusters, std::vector<bool> borderNodes) {
		std::vector<bool> hsn(vertices.size());
		std::vector<std::vector<Vertex<T>*>> predecessorVector(vertices.size());
		for (int s = 0; s < vertices.size(); s++) {
			if (!borderNodes[s]) continue;
			std::fill(predecessorVector.begin(), predecessorVector.end(), std::vector<Vertex<T>*>());
			localBFSAllPaths(s, clusters, predecessorVector);
			for (int t = s + 1; t < vertices.size(); t++) {
				if (!borderNodes[t] || clusters[s] != clusters[t]) continue;
				std::vector<std::vector<Vertex<T>*>> paths;
				//paths.push_back({ &(vertices[t]) });
				//if (s == 19 && t == 28)
				//	std::cout << "This is the bad pair";
				collectPaths(s, t, paths, predecessorVector);
				for (int i = 0; i < paths.size(); i++)
					for (int j = 0; j < paths[i].size(); j++)
						hsn[index(paths[i][j])] = true;
				hsn[t] = true;
			}
			hsn[s] = true;
		}
		/*
		for (int s = 0; s < vertices.size(); s++) {
			if (!borderNodes[s]) continue;
			for (int t = s + 1; t < vertices.size(); t++) {
				if (!borderNodes[t] || clusters[s] != clusters[t]) continue;
				std::vector<std::vector<Vertex<T>*>> paths;
				//paths.push_back({ &(vertices[t]) });
				//if (s == 19 && t == 28)
				//	std::cout << "This is the bad pair";
				collectPaths(s, t, paths, predecessorVectors[s]);
				for (int i = 0; i < paths.size(); i++)
					for (int j = 0; j < paths[i].size(); j++)
						hsn[index(paths[i][j])] = true;
				hsn[t] = true;
			}
			hsn[s] = true;
		}*/
		return hsn;
	}
	std::vector<std::vector<bool>> findExternalNodes(std::vector<bool> hsn, std::vector<int> clusters, std::vector<bool> borderNodes,
		std::vector<std::vector<bool>>& updatedClusters, int nClusters) {
		std::vector<std::vector<bool>> exn(nClusters);
		for (int i = 0; i < nClusters; i++)
			exn[i].resize(vertices.size());
		std::vector<std::vector<Vertex<T>*>> predecessorVector(vertices.size());
		for (int s = 0; s < vertices.size(); s++) {
			updatedClusters[clusters[s]][s] = true;
			if (!borderNodes[s]) continue;
			std::fill(predecessorVector.begin(), predecessorVector.end(), std::vector<Vertex<T>*>());
			std::vector<int> hsnInt(vertices.size());
			for (int i = 0; i < vertices.size(); i++)
				hsnInt[i] = int(hsn[i]);
			localBFSAllPaths(s, hsnInt, predecessorVector);
			if (!borderNodes[s]) continue;
			for (int t = s + 1; t < vertices.size(); t++) {
				if (!borderNodes[t] || clusters[s] != clusters[t]) continue;
				std::vector<std::vector<Vertex<T>*>> paths;
				//paths.push_back({ &(vertices[t]) });
				collectPaths(s, t, paths, predecessorVector);
				for (int i = 0; i < paths.size(); i++)
					for (int j = 1; j < paths[i].size(); j++) {
						if (clusters[index(paths[i][j])] != clusters[s]) {
							exn[clusters[s]][index(paths[i][j])] = true;
							updatedClusters[clusters[s]][index(paths[i][j])] = true;
						}
					}
			}
		}
		return exn;
	}
	void localDeltas(std::vector<int> clusters, std::vector<bool> borderNodes, std::vector<std::vector<bool>> updatedClusters, std::vector<std::vector<bool>> externalNodes,
		std::vector<double>& localDeltas, std::vector<std::vector<double>>& normalisedSigmas, std::vector<std::vector<int>>& distances, int nClusters) {
		//std::vector<double> localDeltas(vertices.size());
		std::vector<std::vector<int>> localSigmas(vertices.size());
		for (int s = 0; s < vertices.size(); s++) {
			std::vector<double> deltas(vertices.size());
			std::vector<int> sigmas(vertices.size());
			std::vector<std::vector<Vertex<T>*>> predecessors(vertices.size());
			distances[s].resize(vertices.size());
			localSigmas[s].resize(vertices.size());
			deltas[s] = 1.0;
			sigmas[s] = 1;
			//localDeltas[s] = 1.0;
			//normalisedSigmas[s] = 1;
			std::vector<int> updatedClustersInt(vertices.size());
			for (int i = 0; i < vertices.size(); i++) {
				updatedClustersInt[i] = int(updatedClusters[clusters[s]][i]);
			}
			modifiedBrandesV1BFS(s, updatedClustersInt, predecessors, deltas, sigmas, localDeltas, localSigmas[s], distances[s], externalNodes[clusters[s]]); // !
			//std::cout << "";
		}

		for (int s = 0; s < vertices.size(); s++) {
			int minDistance = _I32_MAX;
			int minSigma = _I32_MAX;
			for (int v = 0; v < vertices.size(); v++) {
				if (clusters[v] != clusters[s] || !borderNodes[v]) continue;
				if (distances[v][s] < minDistance) minDistance = distances[v][s];
				if (localSigmas[v][s] < minSigma) minSigma = localSigmas[v][s];
			}
			for (int v = 0; v < vertices.size(); v++) {
				if (clusters[v] != clusters[s] || !borderNodes[v]) continue;
				normalisedSigmas[v].resize(vertices.size(), 0);
				//distances[v].resize(vertices.size(), 0);
				distances[v][s] -= minDistance;
				normalisedSigmas[v][s] = double(localSigmas[v][s]) / minSigma;
			}
			localDeltas[s] /= 2;
		}
	}
	struct equivalenceClass {
		std::vector<int> indices;
		std::vector<int> distances;
		std::vector<double> sigmas;
		equivalenceClass(int vertex, std::vector<int> _distances, std::vector<double> _sigmas) {
			indices.push_back(vertex);
			distances = _distances;
			sigmas = _sigmas;
		}
		void addVertex(int vertex) {
			indices.push_back(vertex);
		}
		int random() {
			return indices[rand() % indices.size()];
		}
	};
	std::vector<std::vector<equivalenceClass>> findClasses(std::vector<std::vector<int>> distances, std::vector<std::vector<double>> sigmas,
		std::vector<int> clusters, std::vector<bool> borderNodes, int nClusters) {
		std::vector<std::vector<equivalenceClass>> classes;
		for (int i = 0; i < nClusters; i++) {
			std::vector<equivalenceClass> clusterClasses;
			for (int s = 0; s < vertices.size(); s++) {
				if (clusters[s] != i) continue;
				bool foundClass = false;
				std::vector<int> BNdists;
				std::vector<double> BNsigmas;
				for (int v = 0; v < vertices.size(); v++) {
					if (clusters[v] != i || !borderNodes[v]) continue;
					BNdists.push_back(distances[v][s]);
					BNsigmas.push_back(sigmas[v][s]);
				}
				for (int j = 0; j < clusterClasses.size(); j++) {
					//BNdists = {};
					//BNsigmas = {};
					bool suitableClass = true;
					int nNode = 0;
					for (int v = 0; v < vertices.size(); v++) {
						if (clusters[v] != i || !borderNodes[v]) continue;
						if (distances[v][s] != clusterClasses[j].distances[nNode] || sigmas[v][s] != clusterClasses[j].sigmas[nNode]) suitableClass = false;
						nNode++;
					}
					if (suitableClass) {
						clusterClasses[j].addVertex(s);
						foundClass = true;
					}
				}
				if (!foundClass)
					clusterClasses.push_back(equivalenceClass(s, BNdists, BNsigmas));
			}
			classes.push_back(clusterClasses);
		}
		return classes;
	}
	std::vector<double> globalDeltas(std::vector<std::vector<equivalenceClass>> classes, std::vector<int> clusters, int nClusters) {
		std::vector<double> res(vertices.size());
		for (int c = 0; c < nClusters; c++) {
			for (int i = 0; i < classes[c].size(); i++) {
				int pivot = classes[c][i].random();
				std::vector<std::vector<Vertex<T>*>> predecessors(vertices.size());
				std::vector<double> deltaContribs(vertices.size());
				modifiedBrandesV2BFS(pivot, clusters, predecessors, deltaContribs, nClusters);
				for (int s = 0; s < vertices.size(); s++) {
					res[s] += deltaContribs[s] * classes[c][i].indices.size() / 2;
				}
			}
		}
		return res;
	}
	void DFS(Vertex<T>* current, std::vector<bool>& markedVertices, std::vector<Vertex<T>*>& path, std::vector<int>& components, int currentComponent) {
		std::stack<Vertex<T>*> st;
		st.push(current);
		while (!st.empty()) {
			Vertex<T>* u = st.top();
			st.pop();
			path.push_back(u);
			markedVertices[index(u)] = true;
			components[index(u)] = currentComponent;
			for (int i = 0; i < u->adjacencyList.size(); i++) {
				Vertex<T>* it = u->adjacencyList[i];
				if (markedVertices[index(it)]) continue;
				st.push(it);
			}
		}
		return;
	}
	void dijkstra(Vertex<T>* current, std::vector<bool>& markedVertices, std::vector<int>& distances, std::vector<Vertex<T>*>& predecessors, std::set<VertexDistance<T>>& s) {

		VertexDistance<T> vd1(current, 0);
		s.insert(vd1);

		while (!s.empty()) {

			current = s.begin()->vertex;
			s.erase(s.begin());

			for (int i = 0; i < current->adjacencyList.size(); i++) {
				Vertex<T>* it = current->adjacencyList[i];
				if (markedVertices[index(it)]) continue;
				if (distances[index(current)] + current->weights[i] < distances[index(it)] || distances[index(it)] == UNREACHABLE) {
					distances[index(it)] = distances[index(current)] + current->weights[i];
					predecessors[index(it)] = current;
					VertexDistance<T> vd2(&(vertices[index(it)]), distances[index(it)]);
					s.insert(vd2);
				}
				markedVertices[index(current)] = true;
			}
		}

	}
	void initDSU(std::vector<int>& p, std::vector<int>& rk) {
		for (int i = 0; i < p.size(); i++) {
			p[i] = i;
			rk[i] = 1;
		}
	}
	int getRoot(std::vector<int>& p, std::vector<int>& rk, int v) {
		if (p[v] == v) return v;
		else return p[v] = getRoot(p, rk, p[v]);
	}
	bool merge(std::vector<int>& p, std::vector<int>& rk, int a, int b) {
		int ra = getRoot(p, rk, a), rb = getRoot(p, rk, b);
		if (ra == rb) return false;
		else if (rk[ra] < rk[rb]) p[ra] = rb;
		else if (rk[rb] < rk[ra]) p[rb] = ra;
		else {
			p[ra] = rb;
			rk[rb]++;
		}
		return true;
	}
public:
	std::vector<Vertex<T>> vertices;
	//int** weights;
	//bool weightsAllocated = false;
	Graph() {};
	Graph(std::vector<Vertex<T>> v) {
		vertices = v;
	}
	//Graph(std::vector<Vertex<T>> v, int** ws) {
	//	vertices = v;
	//	weights = ws;
	//}
	Graph(const Graph<T>& g) {
		vertices = {};

		for (int i = 0; i < g.vertices.size(); i++) {
			vertices.push_back(Vertex<int>(Pair<int>(g.vertices[i].data.first, g.vertices[i].data.second), {}, g.vertices[i].weights));
		}


		for (int i = 0; i < g.vertices.size(); i++) {
			for (int j = 0; j < g.vertices[i].adjacencyList.size(); j++) {
				vertices[i].adjacencyList.push_back(&(vertices[g.index(g.vertices[i].adjacencyList[j])]));
			}
		}
	}
	~Graph() {
	}
	Graph<T> operator=(const Graph<T>& g) {
		vertices = {};

		for (int i = 0; i < g.vertices.size(); i++) {
			vertices.push_back(Vertex<int>(Pair<int>(g.vertices[i].data.first, g.vertices[i].data.second), {}, g.vertices[i].weights));
		}


		for (int i = 0; i < g.vertices.size(); i++) {
			for (int j = 0; j < g.vertices[i].adjacencyList.size(); j++) {
				vertices[i].adjacencyList.push_back(&(vertices[g.index(g.vertices[i].adjacencyList[j])]));
			}
		}
		return *this;
	}
	int weight(int v1, int v2) {
		for (int i = 0; i < vertices[v1].adjacencyList.size(); i++) {
			if (index(vertices[v1].adjacencyList[i]) == v2) return vertices[v1].weights[i];
		}
		return 0;
	}
	int index(Vertex<T>* pvx) const {
		int index = pvx - &(vertices[0]);
		return index;
	}

	std::vector<Vertex<T>*> BFS(int startingVertex, std::vector<int>* d = nullptr, std::vector<Vertex<T>*>* pred = nullptr, double a = 1, double b = 1) {
		std::vector<bool> markedVertices(vertices.size());
		markedVertices[startingVertex] = true;
		std::vector<Vertex<T>*> path = {};
		path.push_back(&(vertices[startingVertex]));
		std::vector<int> distances(vertices.size());
		for (int i = 0; i < vertices.size(); i++) {
			if (i == startingVertex) distances[i] = 0;
			else distances[i] = UNREACHABLE;
		}
		std::vector<Vertex<T>*> predecessors(vertices.size());
		std::queue<Vertex<T>*> q;
		q.push(&(vertices[startingVertex]));
		std::vector<bool> frontier(vertices.size());
		frontier[startingVertex] = true;
		int i = 0;
		while (!q.empty() && std::any_of(frontier.begin(), frontier.end(), [](bool v) {return v; })) {
			BFS(&(vertices[startingVertex]), markedVertices, path, distances, predecessors, q, a);
			i++;
			if (i == 2) break;
			if (!q.empty() && std::any_of(frontier.begin(), frontier.end(), [](bool v) {return v; })) {
				startingVertex = index(path[path.size() - 1]);
				//std::cout << "->";
				convertQueueToBitset(q, frontier);
				BFSBottomUp(&(vertices[startingVertex]), markedVertices, path, distances, predecessors, frontier, b);
				startingVertex = index(path[path.size() - 1]);
				//std::cout << "<-";
				convertBitsetToQueue(frontier, q);
			}
		}
		if (d != nullptr)
			*d = distances;
		if (pred != nullptr)
			*pred = predecessors;
		return path;
	}

	void BFSAllPaths(int startingVertex, std::vector<std::vector<std::vector<Vertex<T>*>>>& _paths) {
		//std::vector<bool> markedVertices(vertices.size());
		std::vector<std::vector<Vertex<T>*>> predecessors(vertices.size());
		std::vector<std::vector<std::vector<Vertex<T>*>>> paths(vertices.size());
		//markedVertices[startingVertex] = true;
		std::vector<Vertex<T>*> path = {};
		path.push_back(&(vertices[startingVertex]));
		std::vector<int> distances(vertices.size());
		for (int i = 0; i < vertices.size(); i++) {
			if (i == startingVertex) distances[i] = 0;
			else distances[i] = UNREACHABLE;
		}
		std::queue<Vertex<T>*> q;
		q.push(&(vertices[startingVertex]));
		while (!q.empty()) { //
			Vertex<T>* u = q.front();
			q.pop();
			for (int i = 0; i < u->adjacencyList.size(); i++) {
				Vertex<T>* it = u->adjacencyList[i];
				if (distances[index(it)] == UNREACHABLE) {
					//markedVertices[index(it)] = true;
					//path.push_back(it);
					distances[index(it)] = distances[index(u)] + 1;
					//predecessors[index(it)] = u;
					q.push(it);
				}
				if (distances[index(u)] + 1 == distances[index(it)]) {
					predecessors[index(it)].push_back(u);
				}
			}
		} //
		paths[startingVertex].push_back({});
		for (int i = 0; i < vertices.size(); i++) {
			if (i == startingVertex) continue;
			paths[i].push_back({ &(vertices[i]) });
			std::vector<int> currPaths = { 0 };
			collectPaths(startingVertex, i, paths[i], predecessors);
		}
		_paths = paths;
	}

	std::vector<Vertex<T>*> DFS(int startingVertex = 0, std::vector<int>* comp = nullptr, int* nComponents = nullptr) {
		std::vector<bool> markedVertices(vertices.size());
		std::vector<Vertex<T>*> path = {};
		std::vector<int> components(vertices.size());
		int currentComponent = 0;
		DFS(&(vertices[startingVertex]), markedVertices, path, components, currentComponent);
		currentComponent++;
		for (int i = 0; i < vertices.size(); i++) {
			if (!(markedVertices[i])) {
				DFS(&(vertices[i]), markedVertices, path, components, currentComponent);
				currentComponent++;
			}
		}
		if (comp != nullptr)
			*comp = components;
		if (nComponents != nullptr)
			*nComponents = currentComponent;
		return path;
	}
	void dijkstra(int startingVertex, std::vector<int>* d = nullptr, std::vector<Vertex<T>*>* pred = nullptr) {
		std::vector<bool> markedVertices(vertices.size());
		markedVertices[startingVertex] = true;
		std::vector<int> distances(vertices.size());
		for (int i = 0; i < vertices.size(); i++) {
			if (i == startingVertex) distances[i] = 0;
			else distances[i] = UNREACHABLE;
		}
		std::vector<Vertex<T>*> predecessors(vertices.size());
		std::set<VertexDistance<T>> s;
		dijkstra(&(vertices[startingVertex]), markedVertices, distances, predecessors, s);
		if (d != nullptr)
			*d = distances;
		if (pred != nullptr)
			*pred = predecessors;
	}
	Graph<T> prim(int startingVertex = 0) {
		std::vector<bool> markedVertices(vertices.size());
		std::vector<Vertex<T>> vxs(vertices.size());
		Graph<T> res(*this);
		res.vertices = vxs;
		markedVertices[startingVertex] = true;
		while (!std::all_of(markedVertices.begin(), markedVertices.end(), [](bool v) {return v; })) {
			int minWeight = INT_MAX; int v1 = -1, v2 = -1, idx = -1;
			for (int i = 0; i < vertices.size(); i++) {
				if (markedVertices[i]) {
					for (int j = 0; j < vertices[i].adjacencyList.size(); j++) {
						Vertex<T>* it = vertices[i].adjacencyList[j];
						if (markedVertices[index(it)]) continue;
						if (vertices[i].weights[j] < minWeight) {
							minWeight = vertices[i].weights[j];
							v1 = i;
							v2 = index(it);
							idx = j;
						}
					}
				}
			}
			res.addEdge(v1, v2, vertices[v1].weights[idx]);
			markedVertices[v2] = true;
		}
		return res;
	}
	Graph<T> kruskal() {
		std::vector<int> p(vertices.size()), rk(vertices.size());
		std::vector<bool> markedVertices(vertices.size());
		std::vector<EdgeInfo> allEdges;
		for (int i = 0; i < vertices.size(); i++) {
			for (int j = 0; j < vertices[i].adjacencyList.size(); j++) {
				Vertex<T>* it = vertices[i].adjacencyList[j];
				if (index(it) > i) allEdges.push_back({ i, index(it), vertices[i].weights[j] });
			}
		}
		std::sort(allEdges.begin(), allEdges.end());
		std::vector<Vertex<T>> vxs(vertices.size());
		Graph<T> res(*this);
		res.vertices = vxs;
		std::vector<int> components(vertices.size());
		initDSU(p, rk);
		for (int i = 0; i < allEdges.size(); i++)
			if (merge(p, rk, allEdges[i].a, allEdges[i].b))
				res.addEdge(allEdges[i].a, allEdges[i].b, weight(allEdges[i].a, allEdges[i].b));
		return res;
	}
	int heaviestEdge(int& v1, int& v2, int& idx) {
		int maxWeight = 0;
		for (int i = 0; i < vertices.size(); i++)
			for (int j = 0; j < vertices[i].adjacencyList.size(); j++) {
				Vertex<T>* it = vertices[i].adjacencyList[j];
				if (vertices[i].weights[j] > maxWeight) {
					maxWeight = vertices[i].weights[j];
					v1 = i;
					v2 = index(it);
					idx = j;
				}
			}
		return maxWeight;
	}
	std::vector<int> clusterise(int nClusters, int algorithm = 0) {
		Graph<T> mst;
		if (algorithm == 0) mst = prim();
		if (algorithm == 1) mst = kruskal();
		for (int i = 0; i < nClusters - 1; i++) {
			int v1; int v2; int idx;
			mst.heaviestEdge(v1, v2, idx);
			mst.removeEdge(v1, v2);
		}
		std::vector<int> clusters;
		mst.DFS(0, &clusters);

		//for (int i = 0; i < mst.vertices.size(); i++)
		//	delete[] mst.weights[i];
		//delete[] mst.weights;

		return clusters;
	}
	int sumAllWeights(std::vector<int> clusters, int cluster = -1, int vertex = -1, int* incC = nullptr, int* incV = nullptr, int* between = nullptr) {
		int res = 0;
		if (incC != nullptr) *incC = 0;
		if (incV != nullptr) *incV = 0;
		if (between != nullptr) *between = 0;
		if (cluster == -1) {
			clusters = {};
			clusters.resize(vertices.size());
			cluster = 0;
		}
		for (int i = 0; i < vertices.size(); i++) {
			if (i == vertex) {
				for (int j = 0; j < vertices[i].adjacencyList.size(); j++) {
					Vertex<T>* it = vertices[i].adjacencyList[j];
					if (clusters[index(it)] == cluster) {
						if (between != nullptr) *between += vertices[i].weights[j];
					}
					if (incV != nullptr) *incV += vertices[i].weights[j];
				}

			}
			if (clusters[i] == cluster) {
				for (int j = 0; j < vertices[i].adjacencyList.size(); j++) {
					Vertex<T>* it = vertices[i].adjacencyList[j];
					if (clusters[index(it)] == cluster && index(it) <= i) res += vertices[i].weights[j];
					else if (incC != nullptr) *incC += vertices[i].weights[j];
				}
			}
		}
		return res;
	}
	std::vector<int> louvain(int& nClusters, int desiredNClusters = 0) {
		std::vector<int> clusters(vertices.size());
		for (int i = 0; i < vertices.size(); i++)
			clusters[i] = i;
		std::vector<int> communities = clusters;
		nClusters = vertices.size();
		int currNClusters = vertices.size();
		std::vector<Vertex<T>> vxs(vertices.size());

		Graph<T> currNetwork(*this);
		//currNetwork.vertices = vxs;
		double m = sumAllWeights(clusters);
		while (true) {
			//First stage
			bool changed = false;
			std::vector<int> oldClusters = clusters;
			std::vector<bool> communityRemoved(vertices.size());
			for (int i = 0; i < currNetwork.vertices.size(); i++) {
				double maxModularityGain = 0; int argmmg = -1;
				std::vector<bool> clusterProcessed(vertices.size());
				bool singleton = true;
				for (int j = 0; j < currNetwork.vertices[i].adjacencyList.size(); j++) {
					Vertex<T>* it = currNetwork.vertices[i].adjacencyList[j];
					//	std::cout << currNetwork.index(it->data) << std::endl;
					if (clusterProcessed[communities[currNetwork.index(it)]]) continue;
					if (communities[currNetwork.index(it)] == communities[i]) {
						if (currNetwork.index(it) != i) singleton = false;
						continue;
					}
					int sin, stot, ki, kiin;
					sin = currNetwork.sumAllWeights(communities, communities[currNetwork.index(it)], i, &stot, &ki, &kiin);
					double modularityGain = (double(sin + kiin) / (2 * m) - (double(stot + ki) / (2 * m)) * (double(stot + ki) / (2 * m))) -
						(double(sin) / (2 * m) - (double(stot) / (2 * m)) * (double(stot) / (2 * m)) - (double(ki) / (2 * m)) * (double(ki) / (2 * m)));
					if (modularityGain > maxModularityGain) {
						maxModularityGain = modularityGain;
						argmmg = communities[currNetwork.index(it)];
					}
					clusterProcessed[communities[currNetwork.index(it)]] = true;
				}
				if (argmmg != -1) {
					if (singleton) {
						currNClusters--;
						communityRemoved[i] = true; //
					}
					for (int j = 0; j < vertices.size(); j++)
						if (oldClusters[j] == communities[i]) clusters[j] = argmmg;
					communities[i] = argmmg;
					changed = true;
				}
			}
			//Second stage
			if (!changed) return clusters;
			nClusters = currNClusters;
			////int** newws = new int* [nClusters];
			//for (int i = 0; i < nClusters; i++) {
			//	newws[i] = new int[nClusters];
			//}
			std::vector<Vertex<T>> newvxs(nClusters);
			for (int i = 0; i < nClusters; i++)
				newvxs[i].data.first = i;
			Graph newNetwork(newvxs);

			std::vector<int> newCommunities(vertices.size());
			std::vector<bool> communityHadBeenRemoved = communityRemoved;
			std::vector<bool> oldCommunityProcessed(vertices.size());
			std::vector<int> transformTable(vertices.size());
			std::fill(transformTable.begin(), transformTable.end(), -1);
			int currCommunity = 0;

			for (int i = 0; i < currNetwork.vertices.size(); i++) {
				currCommunity = communities[i];
				int sin, kiin;
				if (communities[i] >= nClusters)
					if (transformTable[communities[i]] != -1) currCommunity = transformTable[communities[i]];
				if (!oldCommunityProcessed[communities[i]]) {
					if (communities[i] >= nClusters) {
						if (transformTable[communities[i]] != -1) currCommunity = transformTable[communities[i]];
						else {
							bool foundSlot = false;
							for (int j = 0; j < nClusters; j++)
								if (communityRemoved[j]) {
									currCommunity = j;
									communityRemoved[j] = false;
									transformTable[communities[i]] = j;
									foundSlot = true;
									break;
								}
							if (!foundSlot)
								std::cout << "Slot not found";
						}
					}
					sin = sumAllWeights(communities, currCommunity);
					newNetwork.addEdge(currCommunity, currCommunity, sin);
				}
				for (int k = 0; k < currNetwork.vertices[i].adjacencyList.size(); k++) {
					Vertex<T>* it = currNetwork.vertices[i].adjacencyList[k];
					if (communities[currNetwork.index(it)] == communities[i]) continue;
					int otherCommunity = communities[currNetwork.index(it)];
					if (communities[currNetwork.index(it)] >= nClusters) {
						if (transformTable[communities[currNetwork.index(it)]] != -1) otherCommunity = transformTable[communities[currNetwork.index(it)]];
						else {
							bool foundSlot = false;
							for (int j = 0; j < nClusters; j++)
								if (communityRemoved[j]) {
									otherCommunity = j;
									communityRemoved[j] = false;
									transformTable[communities[currNetwork.index(it)]] = j;
									foundSlot = true;
									break;
								}
							if (!foundSlot)
								std::cout << "Slot not found";
						}
					}
					if (!newNetwork.areNeighbours(currCommunity, otherCommunity)) {
						newNetwork.addEdge(currCommunity, otherCommunity, 0);
						//newNetwork.weights[currCommunity][otherCommunity] = 0;
					}
					newNetwork.assignWeight(currCommunity, otherCommunity, newNetwork.weight(currCommunity, otherCommunity) + currNetwork.weight(i, currNetwork.index(it))); // !
					//foundLink[otherCommunity] = true;
					//if(!currCommuityProcessed && ) newNetwork.addEdge(currCommunity, , 1)
					//currNetwork.weights[currCommunity][
				}
				//if (!oldCommunityProcessed[communities[i]]) {
				//	newNetwork.weights[currCommunity][currCommunity] = sin;
				//}
				oldCommunityProcessed[communities[i]] = true;
			}

			for (int i = 0; i < nClusters; i++)
				communities[i] = i;
			for (int i = 0; i < vertices.size(); i++)
				if (clusters[i] >= nClusters) clusters[i] = transformTable[clusters[i]];

			currNetwork.vertices = {};
			for (int i = 0; i < newNetwork.vertices.size(); i++) {
				currNetwork.vertices.push_back(Vertex<int>(Pair<int>(newNetwork.vertices[i].data.first, newNetwork.vertices[i].data.second),
					{}, newNetwork.vertices[i].weights));
			}
			for (int i = 0; i < newNetwork.vertices.size(); i++) {
				for (int j = 0; j < newNetwork.vertices[i].adjacencyList.size(); j++)
					currNetwork.vertices[i].adjacencyList.push_back(&(currNetwork.vertices[newNetwork.index(newNetwork.vertices[i].adjacencyList[j])]));
			}
			if (nClusters < desiredNClusters) return clusters;
			/*
			for (int i = 0; i < newNetwork.vertices.size(); i++) {
				for (typename List<Vertex<T>*>::iterator it = newNetwork.vertices[i].adjacencyList.begin(); it != newNetwork.vertices[i].adjacencyList.end(); ++it) {
					currNetwork.vertices[i].adjacencyList.push_front(&(currNetwork.vertices[newNetwork.index(it)]));
				}
			}

			for (int i = 0; i < nClusters; i++) {
				for (int j = 0; j < nClusters; j++) {
					currNetwork.weights[i][j] = newNetwork.weights[i][j];
				}
				delete[] newNetwork.weights[i];
			}
			delete[] newNetwork.weights;
			*/
		}


		return clusters;
	}
	//double modularity(std::vector<int> communities) {
	//	double m = sumAllWeights(clusters);

	//}
	void adoptSingletons(std::vector<int>& clusters, int& nClusters) {
		for (int i = 0; i < vertices.size(); i++) {
			bool singleton = true;
			for (int j = 0; j < vertices[i].adjacencyList.size(); j++)
				if (clusters[i] == clusters[index(vertices[i].adjacencyList[j])])
					singleton = false;
			if (singleton)
				clusters[i] = clusters[index(vertices[i].adjacencyList[0])];
		}
		std::vector<bool> slotFilled;
		for (int i = 0; i < nClusters; i++) {
			if (std::find(clusters.begin(), clusters.end(), i) != clusters.end()) slotFilled.push_back(true);
			else slotFilled.push_back(false);
		}
		for (int i = nClusters - 1; i >= 0; i--) {
			if (slotFilled[i]) {
				bool slotFound = false;
				for (int j = 0; j < nClusters; j++)
					if (!slotFilled[j]) {
						for (int k = 0; k < vertices.size(); k++)
							if (clusters[k] == i) clusters[k] = j;
						slotFilled[j] = true;
						slotFound = true;
						break;
					}
				if (!slotFound) break;
			}
			nClusters--;
		}
	}
	std::vector<double> brandes(std::vector<int> clusters = {}) {
		if (clusters.size() == 0) {
			for (int i = 0; i < vertices.size(); i++)
				clusters.push_back(0);
		}
		std::vector<double> res(vertices.size());
		//std::vector<std::vector<int>> S(vertices.size(), vector<int>(vertices.size(), 0));
		for (int i = 0; i < vertices.size(); i++) {
			std::vector<double> deltas(vertices.size());
			std::vector<int> sigmas(vertices.size());
			std::vector<std::vector<Vertex<T>*>> predecessors(vertices.size());
			deltas[i] = 1.0;
			sigmas[i] = 1;
			brandesBFS(i, clusters, predecessors, deltas, sigmas, res);
		}
		for (int i = 0; i < vertices.size(); i++)
			res[i] = res[i] / 2;
		return res;
	}
	std::vector<double> brandesNaive() {
		std::vector<double> res(vertices.size());
		//std::vector<std::vector<int>> S(vertices.size(), vector<int>(vertices.size(), 0));
		std::vector<std::vector<std::vector<std::vector<Vertex<T>*>>>> paths(vertices.size());
		for (int i = 0; i < vertices.size(); i++) {
			BFSAllPaths(i, paths[i]);
		}
		for (int v = 0; v < vertices.size(); v++) {
			for (int s = 0; s < vertices.size(); s++) {
				if (s == v) continue;
				for (int t = s + 1; t < vertices.size(); t++) {
					int sigmaV = 0;
					if (t == v) continue;
					for (int i = 0; i < paths[s][t].size(); i++)
						for (int j = 0; j < paths[s][t][i].size(); j++)
							if (index(paths[s][t][i][j]) == v) sigmaV++;
					res[v] += double(sigmaV) / paths[s][t].size();
				}
			}
		}
		return res;
	}
	std::vector<double> fastBC() {
		std::vector<double> res(vertices.size());
		int nClusters;
		//int dnc = pow(vertices.size(), 0.5);
		std::vector<int> clusters = louvain(nClusters);
		adoptSingletons(clusters, nClusters);

		for (int i = 0; i < nClusters; i++) {
			std::cout << i << ": " << std::count(clusters.begin(), clusters.end(), i) << std::endl;
		}

		std::vector<bool> borderNodes = findBorderNodes(clusters);
		std::vector<bool> hsn = buildHSN(clusters, borderNodes);
		std::vector<std::vector<bool>> updatedClusters(nClusters);
		for (int i = 0; i < nClusters; i++)
			updatedClusters[i].resize(vertices.size());
		std::vector<std::vector<bool>> externalNodes = findExternalNodes(hsn, clusters, borderNodes, updatedClusters, nClusters);
		std::vector<double> locDeltas(vertices.size());
		std::vector<std::vector<int>> distances(vertices.size());
		std::vector<std::vector<double>> normalisedSigmas(vertices.size());
		localDeltas(clusters, borderNodes, updatedClusters, externalNodes, locDeltas, normalisedSigmas, distances, nClusters);
		std::vector<std::vector<equivalenceClass>> classes = findClasses(distances, normalisedSigmas, clusters, borderNodes, nClusters);
		std::vector<double> globDeltas = globalDeltas(classes, clusters, nClusters);
		for (int v = 0; v < vertices.size(); v++) {
			res[v] += locDeltas[v];
			res[v] += globDeltas[v];
		}
		return res;
	}
	void addEdge(int v1, int v2, int weight = 0) {
		vertices[v1].adjacencyList.push_back(&(vertices[v2]));
		if (v1 != v2)
			vertices[v2].adjacencyList.push_back(&(vertices[v1]));
		vertices[v1].weights.push_back(weight);
		if (v1 != v2)
			vertices[v2].weights.push_back(weight);
	}
	void assignWeight(int v1, int v2, int weight) {
		for (int i = 0; i < vertices[v1].adjacencyList.size(); i++)
			if (index(vertices[v1].adjacencyList[i]) == v2) {
				vertices[v1].weights[i] = weight;
				break;
			}
		for (int i = 0; i < vertices[v2].adjacencyList.size(); i++)
			if (index(vertices[v2].adjacencyList[i]) == v1) {
				vertices[v2].weights[i] = weight;
				break;
			}
	}
	bool areNeighbours(int v1, int v2) {
		for (int i = 0; i < vertices[v1].adjacencyList.size(); i++)
			if (index(vertices[v1].adjacencyList[i]) == v2)
				return true;
		return false;
	}
	void removeEdge(int v1, int v2) {
		for (int i = 0; i < vertices[v1].adjacencyList.size(); i++)
			if (index(vertices[v1].adjacencyList[i]) == v2) {
				vertices[v1].adjacencyList.erase(vertices[v1].adjacencyList.begin() + i);
				vertices[v1].weights.erase(vertices[v1].weights.begin() + i);
				break;
			}
		for (int i = 0; i < vertices[v2].adjacencyList.size(); i++)
			if (index(vertices[v2].adjacencyList[i]) == v1) {
				vertices[v2].adjacencyList.erase(vertices[v2].adjacencyList.begin() + i);
				vertices[v2].weights.erase(vertices[v2].weights.begin() + i);
				break;
			}
	}
};

Graph<int> generateGraph(int nVertices, double chance) {
	const static int maxWeight = 1;
	std::vector<Vertex<int>> vertices(nVertices);
	/*
	weights = new int* [nVertices];
	for (int i = 0; i < nVertices; i++) {
		weights[i] = new int[nVertices];
		for (int j = 0; j < nVertices; j++)
			weights[i][j] = 0;
	}*/
	Graph<int> graph(vertices);
	for (int i = 0; i < nVertices; i++) {
		for (int j = i + 1; j < nVertices; j++) {
			int rN = rand();
			if (rN < RAND_MAX * chance) {
				graph.addEdge(i, j, 1);
			}
		}
		graph.vertices[i].data.first = i;
	}
	return graph;
}

Graph<int> washington(int N, int**& weights) {
	const int nVertices = 3 * N + 3;
	std::vector<Vertex<int>> vertices(nVertices);
	/*
	weights = new int* [nVertices];
	for (int i = 0; i < nVertices; i++) {
		weights[i] = new int[nVertices];
		for (int j = 0; j < nVertices; j++)
			weights[i][j] = 0;
	}*/

	Graph<int> graph(vertices);
	for (int i = 0; i < 3 * N + 3; i++) {
		Vertex<int> vx(Pair<int>(i, N), {}, {});
		graph.vertices[i] = vx;
	}
	graph.addEdge(0, 1, N);
	int currEdge = 3;
	for (int i = 0; i < N; i++) {
		graph.addEdge(1, currEdge, N);
		currEdge++;
		graph.addEdge(currEdge - 1, currEdge, N);
		currEdge++;
		graph.addEdge(currEdge - 1, 2, N);
	}
	graph.addEdge(2, currEdge, N);
	currEdge++;
	for (int i = 0; i < N - 1; i++) {
		graph.addEdge(currEdge - 1, currEdge, N);
		currEdge++;
	}
	return graph;
}

/*
template<class T> Graph<int> graphFromPoints(std::vector<T> pts, int**& weights, double(*distance)(T, T)) {
	const int nVertices = pts.size();
	std::vector<Vertex<int>> vertices(nVertices);
	weights = new int* [nVertices];
	for (int i = 0; i < nVertices; i++) {
		weights[i] = new int[nVertices];
		for (int j = 0; j < nVertices; j++)
			weights[i][j] = 0;
	}
	Graph<int> graph(vertices, weights);
	for(int i = 0; i < nVertices; i++)
		for (int j = 0; j < nVertices; j++) {
			graph.addEdge(i, j, int(distance(pts[i], pts[j]) * 100));
		}
	return graph;
}
*/

Graph<int> graphFromEdges(int nVertices, std::vector<std::vector<int>> edges, int**& weights) {
	std::vector<Vertex<int>> vertices(nVertices);
	//weights = new int* [nVertices];
	for (int i = 0; i < nVertices; i++) {
		//weights[i] = new int[nVertices];
		vertices[i].data.first = i;
		//for (int j = 0; j < nVertices; j++)
		//	weights[i][j] = 1;
	}
	Graph<int> graph(vertices);
	for (int i = 0; i < edges.size(); i++) {
		graph.addEdge(edges[i][0], edges[i][1], 1);
	}
	return graph;
}

/**
Graph<int> graphFromMM(std::string path) {
	std::fstream fs(path);
	int v1, v2, weight;
	std::string line;
	do {
		fs >> line;
	} while (line[0] == '%');
}
*/
std::vector<std::vector<int>> edges = { {0,1}, {0,2}, {6,2}, {1,5}, {2,3}, {2,4}, {1,3}, {1,6} };


const int ITERATIONS = 1;
const int NVERTICES = ITERATIONS * 9;
//const int NVERTICES = 10;

const double a = 0.0001, b = 100;

int main()
{
	//srand(time(NULL));
	srand(115114);

	std::vector<Vertex<int>> vertices;

	for (int i = 0; i < NVERTICES; i++) {
		vertices.push_back(Vertex<int>(Pair<int>(i, i * 12), {}, {}));
	}

	// srand(time(NULL))
	int n = 21000;
	std::cout << std::endl << "n = " << n << std::endl;
	//Graph<int> graph = graphFromEdges(7, edges, weights);
	Graph<int> graph = generateGraph(n, 3.0 / n);


	std::vector<int> comp; int nComp;
	graph.DFS(0, &comp, &nComp);
	//sum = 0;
	std::vector<bool> componentProcessed(nComp);
	componentProcessed[0] = true;
	for (int i = 0; i < n; i++) {
		if (componentProcessed[comp[i]]) continue;
		int rV;
		do {
			rV = rand() % n;
		} while (comp[rV] != comp[i] - 1);
		graph.addEdge(i, rV, 1);
		componentProcessed[comp[i]] = true;

	}

	clock_t begin_time;

	int nClusters;
	begin_time = clock();
	graph.louvain(nClusters);

	std::cout << "Louvain: " << float(clock() - begin_time) / CLOCKS_PER_SEC << "s, " << nClusters << " clusters" << std::endl;

	begin_time = clock();
	graph.clusterise(nClusters);

	std::cout << "Prim: " << float(clock() - begin_time) / CLOCKS_PER_SEC << std::endl;
	char ch; std::cin >> ch;
	//graph.DFS(0, &comp);
	//std::cout << "c";
	//Graph<int> graph1 = washington(3000, weights);

	 // DANIEL
	begin_time = clock();
	std::vector<double> bcsf = graph.fastBC();
	std::cout << "Daniel: " << float(clock() - begin_time) / CLOCKS_PER_SEC << std::endl;
	begin_time = clock();
	std::vector<double> bcs = graph.brandes();
	std::cout << "Brandes: " << float(clock() - begin_time) / CLOCKS_PER_SEC << std::endl;
	//for (int i = 0; i < bcs.size(); i++) {
	//	std::cout << bcs[i] << std::endl;
	//}

	//std::vector<double> bcs = graph.brandes();
	//std::cout << float(clock() - begin_time) / CLOCKS_PER_SEC;

	//std::vector<double> bcsn = graph.brandesNaive();
	double maxDiff = -1; int idx = -1; double bcsidx = -1, bcsnidx = -1;

	//<< " " << bcsn[i] << std::endl;
	for (int i = 0; i < bcs.size(); i++)

		if (abs(bcs[i] - bcsf[i]) > maxDiff) {
			//std::cout << "i = " << i << ", Brandes BC is " << bcs[i] << ", naive BC is " << bcsf[i] << std::endl;
			idx = i;
			bcsidx = bcs[i];
			bcsnidx = bcsf[i];
			maxDiff = abs(bcs[i] - bcsf[i]);
		}
	std::cout << "Biggest difference is " << abs(bcsidx - bcsnidx) << " at i = " << idx << ", Brandes BC is " << bcsidx << ", Daniel BC is " << bcsnidx << std::endl;
	//}

	/*
	std::vector<int> components;
	std::vector<Vertex<int>*> path = graph.DFS(0, &components);
	for (int i = 0; i < path.size(); i++)
		std::cout << path[i]->data.first << " ";
	std::cout << std::endl << std::endl;
	for (int i = 0; i < graph.vertices.size(); i++)
		std::cout << components[i] << " ";
	std::cout << std::endl << std::endl;
	*/
	//std::vector<Iris> irises = readIrisesFromFile("C:/Users/orlov/OneDrive/Documents/Distant/ίθμο/iris.data");
	/*
	for (int i = 0; i < 15; i++) {
		double x = double(rand()) / RAND_MAX, y = double(rand()) / RAND_MAX;
		pts.push_back(Point(x, y));
		std::cout << "Point " << i << " is (" << x << ", " << y << ")\n";
	}
	*/
	//Graph<int> graph = graphFromPoints<Iris>(irises, weights, &irisDistance);
	//std::vector<int> clusters1, clusters2;
	//clusters1 = graph.clusterise(4, 0);
	//clusters2 = graph.clusterise(3, 1);
	//for (int i = 0; i < graph.vertices.size(); i++)
		//std::cout << clusters1[i] << " ";
	//std::cout << std::endl;
	//for (int i = 0; i < graph.vertices.size(); i++)
	//	std::cout << clusters2[i] << " ";
	//std::cout << std::endl << std::endl;
	/*
	std::vector<int> distances;
	std::vector<Vertex<int>*> predecessors;
	std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
	std::vector<Vertex<int>*> BFSpath = graph1.BFS(0, &distances, &predecessors, a, b);
	std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
	std::cout << time_span.count() << " seconds";
	std::cout << std::endl;
	*/

	/*
   for (int i = 0; i < BFSpath.size(); i++)
	   std::cout << BFSpath[i]->data.first << " ";/*
   std::cout << std::endl << std::endl;
   for (int i = 0; i < graph.vertices.size(); i++)
	   std::cout << distances[i] << " ";
   std::cout << std::endl << std::endl;

   graph.dijkstra(1, &distances);
   for (int i = 0; i < graph.vertices.size(); i++)
	   std::cout << distances[i] << " ";
   */
   //for (int i = 0; i < 7; i++)
   //	delete[] weights[i];
   //delete[] weights;
}

