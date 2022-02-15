
angular.module('sudokuApp', [])
.controller('ctl', function($scope) {

    let initializeBoard = (difficulty) => {
        let solution, board, coveredCount;
        [solution, board, coveredCount] = generateBoard(difficulty);
        this.board = board;
        this.solution = solution;
        this.coveredCount = coveredCount;
        this.positions = generatePositions();
        this.actions = {};
        updateErrors();
    }
    initializeBoard.bind(this);

    let clear = () => $('.board .block').removeClass('highlight main');

    let updateErrors = () => {
        let locCounter = {};

        function markLoc(hash) {
            if (!locCounter.hasOwnProperty(hash))
                locCounter[hash] = 0;
            locCounter[hash] += 1
        }

        for (let i = 0; i < 9; i++) {
            for (let j = 0; j < 9; j++) {
                let pos = [i, j];
                if (this.actions[pos] !== undefined)
                    for (let loc of getLocs(i, j, this.actions[pos]))
                        markLoc(loc);

                if (this.board[i][j] !== ' ')
                    for (let loc of getLocs(i, j, this.board[i][j]))
                        markLoc(loc);
            }
        }

        let errors = [];
        for (let i = 0; i < 9; i++) {
            for (let j = 0; j < 9; j++) {
                let pos = [i, j];
                let toAdd = false;
                if (this.actions[pos] !== undefined)
                    for (let loc of getLocs(i, j, this.actions[pos]))
                        toAdd ||= locCounter[loc] >= 2;

                if (this.board[i][j] !== ' ')
                    for (let loc of getLocs(i, j, this.board[i][j]))
                        toAdd ||= locCounter[loc] >= 2;

                if (toAdd) {
                    errors.push(pos);
                }
            }
        }

        $(`.board .block`).removeClass('error');
        for (let pos of errors) {
            [i, j] = pos;
            $(`.board [data-x-pos=${i}][data-y-pos=${j}]`).addClass('error');
        }
        return errors.length
    }
    updateErrors.bind(this);

    document.addEventListener('keyup', (event) => {
        if (event.key === 'Escape') {
            clear()
            return;
        }

        let pos, x, y;
        let cur = $(`.board .block.main`);
        pos = [x, y] = [parseInt(cur.attr('data-x-pos')), parseInt(cur.attr('data-y-pos'))]
        if (pos.includes(undefined))
            return

        if (event.key.startsWith('Arrow')) {
            switch (event.key.slice(5)) {
                case 'Right':
                    pos[1] += 1;
                    break;
                case 'Left':
                    pos[1] -= 1;
                    break;
                case 'Down':
                    pos[0] += 1;
                    break;
                case 'Up':
                    pos[0] -= 1;
                    break;
            }
            [pos[0], pos[1]] = [(pos[0] + 9) % 9, (pos[1] + 9) % 9]
            this.highlight(pos);
        }

        if ('123456789'.includes(event.key)) {
            if (this.board[x][y] !== ' ')
                return
            this.actions[[x, y]] = parseInt(event.key);
            let success = updateErrors() === 0;
            if (Object.keys(this.actions).length === this.coveredCount && success) {
                if(window.confirm('Congrats, you win! Start a new game?'))
                    initializeBoard();
            }
        } else if (event.key === 'Backspace' || event.key === 'Delete'){
            if (this.actions.hasOwnProperty(pos))
                delete this.actions[pos];
            updateErrors();
        }
        $scope.$apply();
    });


    $( "#slider" ).slider({
        min:0,
        max:4,
        value: 2,
        change: function( event, ui ) {
            initializeBoard(4 - ui.value);
            clear();
            $scope.$apply();
        }.bind(this)
    });
    initializeBoard(2);


    this.highlight = function (pos) {
        let x, y;
        [x, y] = pos
        let cur = $(`.board [data-x-pos=${x}][data-y-pos=${y}]`);

        let isSelected = cur.hasClass('main');
        clear();

        // if cur pos is already set, continue no further
        if (isSelected || cur.length === 0)
            return;

        // add highlight
        $(`.board [data-x-pos=${x}], .board [data-y-pos=${y}]`).addClass('highlight');
        cur.addClass('main');
    }
});
