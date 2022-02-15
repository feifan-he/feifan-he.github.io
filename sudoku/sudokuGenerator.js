function randomize(ary) {
    for (let i = 0; i < ary.length; i++) {
        const randIdx = Math.floor(Math.random() * ary.length);
        [ary[i], ary[randIdx]] = [ary[randIdx], ary[i]];
    }
}

function getLocs(i, j, num) {
    return new Set([
        `h,${i},${num}`,
        `v,${j},${num}`,
        `b,${Math.floor(i / 3)},${Math.floor(j / 3)},${num}`]);
}

function remove(i, j, num, seen) {
    const locs = getLocs(i, j, num);
    locs.forEach((x) => seen.delete(x))
}

function try_num(i, j, num, seen) {
    const locs = getLocs(i, j, num);
    if ([...locs].filter((i) => seen.has(i)).length > 0) {
        return false
    }
    locs.forEach((x) => seen.add(x));
    return true;
}

function generate() {
    const board = [];

    // initialize board
    for (let i = 0; i < 9; i++)
        board.push(new Array(9).fill(0));

    const seen = new Set()

    function gen(a = 0) {
        if (a === 9 * 9)
            return true;

        const remaining = [1, 2, 3, 4, 5, 6, 7, 8, 9];
        randomize(remaining);

        const i = Math.floor(a / 9), j = a % 9;
        for (const num of remaining) {
            if (try_num(i, j, num, seen)) {
                board[i][j] = num;
                if (gen(a + 1)) {
                    return true;
                }
                board[i][j] = 0;
                remove(i, j, num, seen)
            }
        }
        return false;
    }

    gen();
    return board;
}

function isUniqueSolution(board) {
    const seen = new Set();
    for (let i = 0; i < 9; i++) {
        for (let j = 0; j < 9; j++) {
            if (board[i][j] !== " ")
                try_num(i, j, board[i][j], seen);
        }
    }

    let combCnt = 0;

    function isUnique(a = 0) {
        if (combCnt > 100)
            return 2

        const i = Math.floor(a / 9), j = a % 9;

        if (a === 9 * 9)
            return 1;

        if (board[i][j] !== " ") {
            return isUnique(a + 1)
        }

        let res = 0;
        for (let k = 1; k <= 9; k++) {
            if (try_num(i, j, k, seen)) {
                res += isUnique(a + 1);
                remove(i, j, k, seen)
                if (res >= 2)
                    return res;
            }
        }
        return res
    }

    return isUnique() === 1;
}

function printBoard(board) {
    for (let i = 0; i < 9; i++) {
        const sections = [];
        for (let j = 0; j < 9; j += 3) {
            sections.push(board[i].slice(j, j + 3).join(' '));
        }
        const row = sections.join(' | ');
        console.log(row)

        if ([2, 5].includes(i)) {
            console.log('-'.repeat(row.length));
        }
    }
    console.log()
}

function coverBoard(board, minVis, maxVis) {
    let res = JSON.parse(JSON.stringify(board));
    let covered = [];
    let numCovered = 0;
    for (let i = 0; i < 9; i++) {
        covered.push([]);
        for (let j = 1; j <= 9; j++) {
            covered[i].push(j);
        }
        randomize(covered[i]);

        let numOfVis = Math.floor(Math.random() * (maxVis - minVis + 1) + minVis);
        for (let j = 0; j < 9 - numOfVis; j++) {
            covered[i].pop();
            numCovered += 1;
        }
    }
    for (let i = 0; i < 9; i++) {
        for (let j = 0; j < 9; j++) {
            let a, b;
            [a, b] = [Math.floor(i / 3), Math.floor(j / 3)];
            let k = a * 3 + b;
            let num = board[i][j];
            if (!covered[k].includes(num))
                res[i][j] = " ";
        }
    }
    return [res, numCovered]
}

function generateBoard(difficulty) {
    if (!(0 <= difficulty && difficulty <= 4))
        throw 'Difficulty should be between 1 to 9'
    let numCovered;
    let isUnique = false;
    let start = performance.now();
    const board = generate();
    let coveredBoard;
    do {
        [coveredBoard, numCovered] = coverBoard(board, Math.max(3 + difficulty, 3), 4 + difficulty)
        isUnique = isUniqueSolution(coveredBoard)
    } while (!isUnique)
    let duration = performance.now() - start
    // console.log('Total Duration: ' + duration)
    // printBoard(board)
    // printBoard(coveredBoard)

    return [board, coveredBoard, numCovered]
}

function generatePositions() {
    const res = [];
    for (let i = 0; i < 9; i++) {
        res.push([]);
        for (let j = 0; j < 9; j++) {
            res[i].push([i, j]);
        }
    }
    return res;
}
