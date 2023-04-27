#include "GraphClass.h"
static const int UNREACHABLE = -1;

VertexDistance::VertexDistance(int pvx, int d) {
	vertex = pvx;
	distance = d;
}
bool VertexDistance::operator<(const VertexDistance vd) const {
	if (distance < vd.distance || (distance == vd.distance && vertex < vd.vertex))
		return true;
	else return false;
}
VertexDistance::VertexDistance(const VertexDistance& vd) {
	vertex = vd.vertex;
	distance = vd.distance;
}

void Graph::convertQueueToBitset(std::queue<int> q, std::vector<bool>& frontier) {
	std::fill(frontier.begin(), frontier.end(), false);
	while (!q.empty()) {
		frontier[(q.front())] = true;
		q.pop();
	}
}
void Graph::convertBitsetToQueue(std::vector<bool> frontier, std::queue<int>& q) {
	while (!q.empty()) {
		q.pop();
	}
	for (int i = 0; i < frontier.size(); i++)
		if (frontier[i])
			q.push(i);
}
void Graph::BFS(int current, std::vector<bool>& markedVertices,
	std::vector<int>& path, std::vector<int>& distances, std::vector<int>& predecessors, std::queue<int>& q,
	double a) {
	bool iterationComplete = false;
	while (true) {
		int u = q.front();
		q.pop();
		for (int i = row_index[u]; i < row_index[u + 1]; i++) {
			int it = adjacencies[i];
			if (!(markedVertices[it])) {
				markedVertices[it] = true;
				path.push_back(it);
				distances[it] = distances[u] + 1;
				predecessors[it] = u;
				q.push(it);
			}
		}
		if (!q.empty() && distances[u] + 1 == distances[q.front()]) {
			std::queue<int> q2 = q;
			int mf = 0, mu = 0;
			while (!q2.empty()) {
				int vx = q2.front();
				q2.pop();
				for (int i = row_index[vx]; i < row_index[vx + 1]; i++) {
					int it = adjacencies[i];
					if (distances[it] == UNREACHABLE || distances[it] == distances[u] + 1)
						mf++;
				}
				for (int i = 0; i < size(); i++)
					if (predecessors[i] == -1) {
						for (int j = row_index[i]; j < row_index[i]; j++) {
							int it = adjacencies[j];
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
void Graph::BFSBottomUp(int current, std::vector<bool>& markedVertices,
	std::vector<int>& path, std::vector<int>& distances, std::vector<int>& predecessors, std::vector<bool>& frontier,
	double b) {
	std::vector<bool> next(size());
	bool iterationComplete = false;
	predecessors[current] = current;
	distances[current] = 0;
	while (std::any_of(frontier.begin(), frontier.end(), [](bool v) {return v; })) {
		for (int i = 0; i < size(); i++)
			if (predecessors[i] == -1) {
				for (int j = row_index[i]; j < row_index[i]; j++) {
					int it = adjacencies[j];
					if (frontier[it]) {
						next[i] = true;
						predecessors[i] = it;
						distances[i] = distances[it] + 1;
						path.push_back(i);
						break;
					}
				}
			}
		frontier = next;
		std::fill(next.begin(), next.end(), false);
		int nf = std::count(frontier.begin(), frontier.end(), true);
		if (nf < size() && iterationComplete) return;
		else iterationComplete = true;
	}
}
void Graph::dijkstra(int current, std::vector<bool>& markedVertices, std::vector<int>& distances, std::vector<int>& predecessors, std::set<VertexDistance>& s) {

	VertexDistance vd1(current, 0);
	s.insert(vd1);

	while (!s.empty()) {

		current = s.begin()->vertex;
		s.erase(s.begin());

		for (int i = row_index[current]; i < row_index[current + 1]; i++) {
			int it = adjacencies[i];
			if (markedVertices[it]) continue;
			if (distances[current] + weights[i] < distances[it] || distances[it] == UNREACHABLE) {
				distances[it] = distances[current] + weights[i];
				predecessors[it] = current;
				VertexDistance vd2(it, distances[it]);
				s.insert(vd2);
			}
			markedVertices[current] = true;
		}
	}

}
void Graph::DFS(int current, std::vector<bool>& markedVertices, std::vector<int>& path, std::vector<int>& components, int currentComponent) {
	std::stack<int> st;
	st.push(current);
	while (!st.empty()) {
		int u = st.top();
		st.pop();
		path.push_back(u);
		markedVertices[u] = true;
		components[u] = currentComponent;
		for (int i = row_index[u]; i < row_index[u + 1]; i++) {
			int it = adjacencies[i];
			if (markedVertices[it]) continue;
			st.push(it);
		}
	}
	return;
}
void Graph::initDSU(std::vector<int>& p, std::vector<int>& rk) {
	for (int i = 0; i < p.size(); i++) {
		p[i] = i;
		rk[i] = 1;
	}
}
int Graph::getRoot(std::vector<int>& p, std::vector<int>& rk, int v) {
	if (p[v] == v) return v;
	else return p[v] = getRoot(p, rk, p[v]);
}
bool Graph::merge(std::vector<int>& p, std::vector<int>& rk, int a, int b) {
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
std::vector<int> Graph::BFS(int startingVertex, std::vector<int>* d, std::vector<int>* pred, double a, double b) {
	std::vector<bool> markedVertices(size());
	markedVertices[startingVertex] = true;
	std::vector<int> path = {};
	path.push_back(startingVertex);
	std::vector<int> distances(size());
	for (int i = 0; i < size(); i++) {
		if (i == startingVertex) distances[i] = 0;
		else distances[i] = UNREACHABLE;
	}
	std::vector<int> predecessors(size());
	std::fill(predecessors.begin(), predecessors.end(), -1);
	std::queue<int> q;
	q.push(startingVertex);
	std::vector<bool> frontier(size());
	frontier[startingVertex] = true;
	int i = 0;
	while (!q.empty() && std::any_of(frontier.begin(), frontier.end(), [](bool v) {return v; })) {
		BFS(startingVertex, markedVertices, path, distances, predecessors, q, a);
		i++;
		if (i == 2) break;
		if (!q.empty() && std::any_of(frontier.begin(), frontier.end(), [](bool v) {return v; })) {
			startingVertex = path[path.size() - 1];
			//std::cout << "->";
			convertQueueToBitset(q, frontier);
			BFSBottomUp(startingVertex, markedVertices, path, distances, predecessors, frontier, b);
			startingVertex = path[path.size() - 1];
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
std::vector<int> Graph::DFS(int startingVertex, std::vector<int>* comp, int* nComponents) {
	std::vector<bool> markedVertices(size());
	std::vector<int> path = {};
	std::vector<int> components(size());
	int currentComponent = 0;
	DFS(startingVertex, markedVertices, path, components, currentComponent);
	currentComponent++;
	for (int i = 0; i < size(); i++) {
		if (!(markedVertices[i])) {
			DFS(i, markedVertices, path, components, currentComponent);
			currentComponent++;
		}
	}
	if (comp != nullptr)
		*comp = components;
	if (nComponents != nullptr)
		*nComponents = currentComponent;
	return path;
}
void Graph::dijkstra(int startingVertex, std::vector<int>* d, std::vector<int>* pred) {
	std::vector<bool> markedVertices(size());
	markedVertices[startingVertex] = true;
	std::vector<int> distances(size());
	for (int i = 0; i < size(); i++) {
		if (i == startingVertex) distances[i] = 0;
		else distances[i] = UNREACHABLE;
	}
	std::vector<int> predecessors(size());
	std::set<VertexDistance> s;
	dijkstra(startingVertex, markedVertices, distances, predecessors, s);
	if (d != nullptr)
		*d = distances;
	if (pred != nullptr)
		*pred = predecessors;
}
Graph Graph::prim(int startingVertex) {
	std::vector<bool> markedVertices(size());
	Graph res(size());
	markedVertices[startingVertex] = true;
	while (!std::all_of(markedVertices.begin(), markedVertices.end(), [](bool v) {return v; })) {
		int minWeight = std::numeric_limits<int>::max(); int v1 = -1, v2 = -1, idx = -1;
		for (int i = 0; i < size(); i++) {
			if (markedVertices[i]) {
				for (int j = row_index[i]; j < row_index[i + 1]; j++) {
					int it = adjacencies[j];
					if (markedVertices[it]) continue;
					if (weights[j] < minWeight) {
						minWeight = weights[j];
						v1 = i;
						v2 = it;
						idx = j;
					}
				}
			}
		}
		res.addEdge(v1, v2, weights[idx]);
		markedVertices[v2] = true;
	}
	return res;
}
Graph Graph::kruskal() {
	std::vector<int> p(size()), rk(size());
	std::vector<bool> markedVertices(size());
	std::vector<EdgeInfo> allEdges;
	for (int i = 0; i < size(); i++) {
		for (int j = row_index[i]; j < row_index[i + 1]; j++) {
			int it = adjacencies[j];
			if (it > i) allEdges.push_back({ i, it, weights[j] });
		}
	}
	std::sort(allEdges.begin(), allEdges.end());
	Graph res(size());
	std::vector<int> components(size());
	initDSU(p, rk);
	for (int i = 0; i < allEdges.size(); i++)
		if (merge(p, rk, allEdges[i].a, allEdges[i].b))
			res.addEdge(allEdges[i].a, allEdges[i].b, weight(allEdges[i].a, allEdges[i].b));
	return res;
}
int Graph::heaviestEdge(int& v1, int& v2, int& idx) {
	int maxWeight = 0;
	for (int i = 0; i < size(); i++)
		for (int j = row_index[i]; j < row_index[i + 1]; j++) {
			int it = adjacencies[j];
			if (weights[j] > maxWeight) {
				maxWeight = weights[j];
				v1 = i;
				v2 = it;
				idx = j;
			}
		}
	return maxWeight;
}
std::vector<int> Graph::clusterise(int nClusters, int algorithm) {
	Graph mst;
	if (algorithm == 0) mst = prim();
	if (algorithm == 1) mst = kruskal();
	for (int i = 0; i < nClusters - 1; i++) {
		int v1; int v2; int idx;
		mst.heaviestEdge(v1, v2, idx);
		mst.removeEdge(v1, v2);
	}
	std::vector<int> clusters;
	mst.DFS(0, &clusters);

	//for (int i = 0; i < mst.size(); i++)
	//	delete[] mst.weights[i];
	//delete[] mst.weights;

	return clusters;
}