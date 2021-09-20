

const numNodes = 25, nodeLength = 34;
const graphWidth = 800, graphHeight = 500;
const gridWith = 15, gridHeight = 11;


function play(replay, wrapper) {

    function _addOrder() {
        wrapper.append($('<div class="play-order">')
            .append(replay.filter((el) => el.hasClass('node'))
                .map((el) => $('<div>').addClass(`ordering ${el.attr('value')}`).text(el.attr('value')))))
    }

    let cur = 0
    function _play() {

        replay[cur].toggleClass('visited')
        if (replay[cur].hasClass('node'))
            wrapper.find(`.ordering.${replay[cur].attr('value')}`).addClass('visited')
        cur = (cur + 1) % replay.length

        if (cur === 0) {
            setTimeout(function () {
                wrapper.find('.visited').removeClass('visited')
            }, 3000)
            return setTimeout(_play.bind(this), 5000)
        }
        setTimeout(_play.bind(this), replay[cur].hasClass('line') ? 300: 700)
    }
    setTimeout(_play.bind(this), 200)
    _addOrder(wrapper, replay)
}

class Node {
    constructor(pos, par=null, left=null, right=null, level=0) {
        this.par = par
        this.pos = pos
        this.left = left;
        this.right = right;
        this.level = level;
    }

    dom(wrapper) {
        return wrapper.find(`.node[value=${this.pos}]`)
    }

    playPreorder(wrapper) {
        const replay = [];

        function _playPreorder(node) {
            if (!node) return
            let edge = wrapper.find(`.line.${node.pos}`)
            if (edge.length) replay.push(edge)
            replay.push(wrapper.find(`.node[value=${node.dom(wrapper).attr("value")}]`))
            _playPreorder(node.left)
            _playPreorder(node.right)
            if (edge.length) replay.push(edge)
        }

        _playPreorder(this)
        play(replay, wrapper)
    }

    playPostorder(wrapper) {
        const replay = [];

        function _playPostorder(node) {
            if (!node) return
            let edge = wrapper.find(`.line.${node.pos}`)
            if (edge.length) replay.push(edge)
            _playPostorder(node.left)
            _playPostorder(node.right)
            replay.push(wrapper.find(`.node[value=${node.dom(wrapper).attr("value")}]`))
            if (edge.length) replay.push(edge)
        }

        _playPostorder(this)
        play(replay, wrapper)
    }

    playInorder(wrapper) {
        const replay = [];

        function _playPostorder(node) {
            if (!node) return
            let edge = wrapper.find(`.line.${node.pos}`)
            if (edge.length) replay.push(edge)
            _playPostorder(node.left)
            replay.push(wrapper.find(`.node[value=${node.dom(wrapper).attr("value")}]`))
            _playPostorder(node.right)
            if (edge.length) replay.push(edge)
        }

        _playPostorder(this)
        play(replay, wrapper)
    }

    playLevelOrder(wrapper) {
        let queue = [this];
        const replay = [];

        while (queue.length) {
            let node = queue.shift()
            let edge = wrapper.find(`.line.${node.pos}`)
            if (edge.length) replay.push(edge)
            replay.push(wrapper.find(`.node[value=${node.dom(wrapper).attr("value")}]`))
            if (edge.length) replay.push(edge)
            if (node.left)
                queue.push(node.left)
            if (node.right)
                queue.push(node.right)
        }

        play(replay, wrapper)
    }
}

class GNode {
    constructor(pos=null, edges=[], x = null, y = null) {
        this.pos = pos
        this.edges = edges
        this.x = x
        this.y = y
    }

    playBFS(wrapper) {
        let visited = new Set()
        let queue = [this]
        let replay = []
        let visitedEdge = new Set()

        while (queue.length) {
            let node = queue.shift()
            if (visited.has(node)) continue
            replay.push(wrapper.find(`.node.${node.pos}`))
            visited.add(node)
            for (let edgeIdx in node.edges) {
                let t = node.edges[edgeIdx]
                let edge1 = `${node.pos}-${t.pos}`
                let edge2 = `${t.pos}-${node.pos}`
                if (!visitedEdge.has(edge1) && !visitedEdge.has(edge2))
                    replay.push(wrapper.find(`.line.${edge1}`))
                visitedEdge.add(edge1)
                queue.push(t)
            }
        }
        play(replay, wrapper)
    }

    playDFS(wrapper) {
        let visited = new Set()
        let replay = []

        function dfs(node) {
            if (visited.has(node.pos))
                return
            visited.add(node.pos)
            replay.push(wrapper.find(`.node.${node.pos}`))

            let edges = node.edges;
            for (let edgeIdx in edges) {
                let t = edges[edgeIdx]
                let notSeen = !visited.has(t.pos)
                if (notSeen)
                    replay.push(wrapper.find(`.line.${node.pos}-${t.pos}`))
                dfs(t)
                if (notSeen)
                    replay.push(wrapper.find(`.line.${node.pos}-${t.pos}`))
            }
        }
        dfs(this)

        play(replay, wrapper)
    }

}

class GridNode {
    constructor(x, y, edges = null) {
        this.x = x
        this.y = y
        this.edges = edges
    }
}

$(document).ready(function () {

    function buildTree(nodes, left, right, par=null, level=0) {
        if (left >= right) return null;
        var rand = level >= 2 ? Math.floor(Math.random() * (right - left)) + left: Math.floor((left + right) / 2)
        return new Node(rand, par,
            buildTree(nodes, left, rand, rand, level + 1),
            buildTree(nodes, rand + 1, right, rand, level + 1),
            level)
    }

    function init(bst, nodes) {
        for (let i = 0; i < numNodes; i++) {
            const node = $('<div>').text('' + i).addClass('node').css({left: 30 * i}).attr('value', i);
            try {
                bst.append(node);
            }catch (e) {
                console.log(e)
            }
            nodes.push(node);
        }
    }

    function drawTree(bst, node, parent=null) {
        if (!node) return

        node.dom(bst).css({top: 60 * (node.level + 1)})
        drawTree(bst, node.left, node)
        drawTree(bst, node.right, node)

        let getPos = (node, dir) => parseInt(node.dom(bst).css(dir)) + (nodeLength / 2)

        if (parent) {
            const x1 = getPos(parent, 'left');
            const y1 = getPos(parent, 'top');
            const x2 = getPos(node, 'left');
            const y2 = getPos(node, 'top');
            bst.append($(`<svg width="${Math.max(x1, x2)}" height="${Math.max(y1, y2)}">
                <line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" class="line-color line ${parent.pos}-${node.pos} ${node.pos}"/></svg>`)
                .addClass('edge'))
        }
    }


    const trees = $('.bst.tree').get();
    for (let idx in trees) {
        if (!trees.hasOwnProperty(idx))
            continue
        const bst = $(trees[idx]);
        const nodes = [];
        init(bst, nodes)
        const root = buildTree(nodes, 0, nodes.length);
        drawTree(bst, root)

        const getLen = (dir) => Math.max(...nodes.map((node) => node.position()[dir] + nodeLength))
        const width = getLen('left')
        const height = getLen('top')
        bst.css({width: width, height: height})

        if (bst.hasClass('pre-order'))
            root.playPreorder(bst)
        else if (bst.hasClass('post-order'))
            root.playPostorder(bst)
        else if (bst.hasClass('in-order'))
            root.playInorder(bst)
        else if (bst.hasClass('level-order'))
            root.playLevelOrder(bst)
    }

    const graphs = $('.graph').get();

    for (let graphIdx in graphs) {
        if (!graphs.hasOwnProperty(graphIdx))
            continue

        const graph = $(graphs[graphIdx]);
        graph.css({width: graphWidth, height: graphHeight})

        let nodes = []

        let dist = (e, f, g, h) => Math.sqrt(Math.pow(e - g, 2) + Math.pow(f - h, 2))



            function connected(nodes) {
                let numConnected = 0;
                var queue = [nodes[0]]
                var visited = new Set()
                while (queue.length) {
                    let node = queue.shift()
                    if (visited.has(node)) continue
                    node.edges.forEach(function (edge) {
                        queue.push(edge)
                    }.bind(this))
                    visited.add(node)
                    numConnected++
                }
                return numConnected === nodes.length
            }

            let isDone = false
            while (!isDone) {

                let cur = 0
                for (let i = 0; i < 1000; i++) {
                    let genPos = (length) => Math.floor(Math.random() * (length - nodeLength))
                    let x = genPos(graphWidth - nodeLength);
                    let y = genPos(graphHeight - nodeLength)

                    let doAdd = true;
                    for (idx in nodes) {
                        let a = nodes[idx].x
                        let b = nodes[idx].y


                        if (dist(a, b, x, y) < 150)
                            doAdd = false;

                    }
                    if (doAdd) {
                        nodes.push(new GNode(cur, [], x, y))
                        cur++
                    }
                }

                let cnt = 0
                while (!connected(nodes) && cnt < 300) {

                    console.log('loop')
                    cnt++
                    var rand = () => Math.floor(nodes.length * Math.random())
                    let a = rand()
                    let b = rand()
                    let skip = false
                    if (!nodes[a].edges.includes(nodes[b]) && a !== b && nodes[a].edges.length < 3 && nodes[b].edges.length < 3) {
                        for (let nodeIdx in nodes) {
                            nodeIdx = parseInt(nodeIdx)
                            if (nodeIdx !== a && nodeIdx !== b) {
                                let n1 = nodes[a]
                                let n2 = nodes[b]
                                let n3 = nodes[nodeIdx]
                                let d1 = dist(n1.x, n1.y, n2.x, n2.y)
                                let d2 = dist(n2.x, n2.y, n3.x, n3.y)
                                let d3 = dist(n1.x, n1.y, n3.x, n3.y)
                                sorted = [d1, d2, d3]
                                sorted.sort()
                                d1 = sorted[0]
                                d2 = sorted[1]
                                d3 = sorted[2]
                                if (Math.abs(d1 + d2 - d3) < d1 * .015) {
                                    skip = true
                                    break
                                }
                            }
                        }
                        if (skip)
                            continue

                        nodes[a].edges.push(nodes[b])
                        nodes[b].edges.push(nodes[a])
                    }
                }
                if (cnt === 300 || nodes.length < 13) {
                    nodes = []
                } else {
                    isDone = true
                }
            }


            for (let nodeIdx in nodes) {
                let node = nodes[nodeIdx]
                node.edges.sort((a, b) => a.pos - b.pos)
                graph.append($('<div>').text(nodeIdx).addClass(`node ${nodeIdx}`).attr('value', nodeIdx).css({left: node.x, top: node.y}))

                node.edges.forEach(function (n2) {
                    let n1 = node
                    let x1 = n1.x, y1 = n1.y, x2 = n2.x, y2 = n2.y;
                    let offset = nodeLength / 2
                    x1 += offset
                    y1 += offset
                    x2 += offset
                    y2 += offset
                    graph.append($(`<svg width="${Math.max(x1, x2)}" height="${Math.max(y1, y2)}" class="edge">
                        <line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" class="line-color line ${n1.pos}-${n2.pos} ${n2.pos}-${n1.pos}"/></svg>`))
                }.bind(this))
            }


        let start = nodes[0]

        if (graph.hasClass('bfs'))
            start.playBFS(graph)
        else if (graph.hasClass('dfs'))
            start.playDFS(graph)

    }


    function drawGrid(wrapper, grid) {
        function drawEdge(x1, y1, x2, y2, classes) {
            let offset = nodeLength / 2 + 4
            x1 += offset
            y1 += offset
            x2 += offset
            y2 += offset
            wrapper.append($(`<svg width="${Math.max(x1, x2)}" height="${Math.max(y1, y2)}" class="edge">
                        <line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}"
                         class="line-color line ${classes}"/></svg>`))
        }

        let cnt = 0
        for (let i = 0; i < grid.length; i++) {
            for (let j = 0; j < grid[0].length; j++) {
                cnt += 1
                nodeLoc = (loc) => loc * 45
                wrapper.append($('<div>').text(cnt)
                    .addClass(`node gridNode ${i}_${j}
                        ${grid[i][j] ? '' : 'obstacle'}`)
                    .css({left: nodeLoc(i), top: nodeLoc(j)}))

                if (i < grid.length - 1) {
                    drawEdge(nodeLoc(i), nodeLoc(j), nodeLoc(i + 1), nodeLoc(j),
                        `${i}_${j}-${i + 1}_${j} ${i + 1}_${j}-${i}_${j}`)
                }

                if (j < grid[0].length - 1) {
                    drawEdge(nodeLoc(i), nodeLoc(j), nodeLoc(i), nodeLoc(j + 1),
                        `${i}_${j}-${i}_${j + 1} ${i}_${j + 1}-${i}_${j}`)
                }

            }
        }
        wrapper.css({width: grid.length * 45 + nodeLength, height: grid[0].length * 45 + nodeLength})
    }

    function initGrid(m, n){
        let grid = [];
        for (let i = 0; i < m; i++) {
            grid.push([])
            for (let j = 0; j < n; j++)
                grid[i].push(false)
        }
        return grid
    }

    function retrieve(grid, m, n, def=false) {
        return (0 <= m && m < grid.length) && (0 <= n && n < grid[0].length)  ? grid[m][n] : def
    }

    function gridDFS(grid, wrapper, x, y) {
        let replay = []
        let visited = new Set()
        function _DFS(a, b) {
            if (visited.has(a + ',' + b)) return
            if (retrieve(grid, a, b)) {
                visited.add(a + ',' + b)
                replay.push(wrapper.find(`.node.${a}_${b}`))
            }

            function visit(c, d) {
                if (retrieve(grid, c, d) && !visited.has(c + ',' + d)) {
                    replay.push(wrapper.find(`.line.${c}_${d}-${a}_${b}`))
                    _DFS(c, d)
                    replay.push(wrapper.find(`.line.${c}_${d}-${a}_${b}`))
                }
            }

            visit(a + 1, b)
            visit(a, b + 1)
            visit(a - 1, b)
            visit(a, b - 1)

            _DFS(a, b)
        }
        _DFS(x, y)
        play(replay, wrapper)
    }

    function gridBFS(grid, wrapper, x, y) {
        let replay = []
        let visited = new Set()
        let lineVisited = new Set()
        let queue = [[x, y]]


        while (queue.length) {
            let loc = queue.shift()
            let a = loc[0]
            let b = loc[1]
            if (visited.has(`${a},${b}`)) continue
            visited.add(`${a},${b}`)
            replay.push(wrapper.find(`.node.${a}_${b}`))

            function visit(c, d) {
                let lineCoords = `${c}_${d}-${a}_${b}`
                let lineCoords2 = `${a}_${b}-${c}_${d}`
                let line = wrapper.find(`.line.${lineCoords}`)
                if (line === null)
                    return

                if (!lineVisited.has(lineCoords) && !lineVisited.has(lineCoords2) && retrieve(grid, c, d)) {
                    lineVisited.add(lineCoords)
                    replay.push(wrapper.find(`.line.${lineCoords}`))
                }

                if (retrieve(grid, c, d) && !visited.has(`${c},${d}`)) {
                    queue.push([c, d])
                }
            }

            visit(a + 1, b)
            visit(a, b + 1)
            visit(a - 1, b)
            visit(a, b - 1)

        }
        play(replay, wrapper)
    }

    const grids = $('.grid').get();
    for (let idx in grids) {
        if (!grids.hasOwnProperty(idx))
            continue
        const wrapper = $(grids[idx]);


        let grid = initGrid(gridWith, gridHeight)

        let midX = Math.floor(gridWith / 2), midY = Math.floor(gridHeight / 2)
        grid[midX][midY] = true
        for (let k = 0, cnt = 0; k < 15 || cnt < 30; k++) {
            cnt = 0
            let newGrid = initGrid(gridWith, gridHeight)
            for (let i = 0; i < gridWith; i++) {
                for (let j = 0; j < gridHeight; j++) {
                    if (grid[i][j]) {
                        newGrid[i][j] = true
                        cnt++
                    } else if ((retrieve(grid, i - 1, j, false) ||
                        retrieve(grid, i, j - 1, false) ||
                        retrieve(grid, i + 1, j, false) ||
                        retrieve(grid, i, j + 1, false)) &&
                        Math.random() < .2
                        ) {
                        newGrid[i][j] = true
                        cnt++
                    }
                }
            }
            grid = newGrid
        }

        drawGrid(wrapper, grid)
        if (wrapper.hasClass('dfs'))
            gridDFS(grid, wrapper, midX, midY)
        else if (wrapper.hasClass('bfs'))
            gridBFS(grid, wrapper, midX, midY)
    }
})