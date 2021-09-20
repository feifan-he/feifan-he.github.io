const xpos = [];
const width = 700, height = 350, numItems = 20, itemWidth = 10;


class Algo {
    constructor(dom) {
        this.dom = dom
        this.initDom = null;
        this.lst = this.randomInts(numItems)
        this.snapshots = []
        this.init()
    }

    init() {

        this.dom.css({width: width, height: height})

        if (this.initDom) {
            this.dom.html(this.initDom)
            return
        }

        for (let i = 0; i < numItems; i++) {
            this.dom.append(
                $(`<div class="item item-${i}">`).css(
                    {
                        height: this.lst[i] * 10 + 10,
                        width: itemWidth,
                        left: width * (i + 2) / (numItems + 3) - (itemWidth / 2)
                    }))
        }
        this.initDom = this.dom.html()
    }

    randomInts(size) {
        const res = [];
        for (let i = 0; i < size; i++)
            res.push(i)

        for (let i = 0; i < size; i++) {
            const rand = Math.floor(Math.random() * size);
            const tmp = res[i];
            res[i] = res[rand]
            res[rand] = tmp
        }
        return res
    }

    swap(i, j, doSwap = true) {
        var lst = this.lst
        if (i === j && doSwap) return

        if (doSwap) {
            var tmp = lst[i]
            lst[i] = lst[j]
            lst[j] = tmp
        }
        this.snapshots.push([i, j, doSwap])
    }

    relabel() {
        // label positions
        const posClass = [];
        this.dom.find('.item').each(function (i, el) {
            var _el = $(el)
            let left = _el.css('left')
            posClass.push([parseInt(left.substr(0, left.length - 2)), _el])
        })
        posClass.sort((m, n) => m[0] < n[0] ? -1 : 1)
        for (var i = 0; i < posClass.length; i++) {
            $(posClass[i][1]).attr('pos', i)
        }
    }

    tog = true
    play(idx = 0) {
        if (idx === this.snapshots.length) {
            this.dom.find('.item').removeClass('highlight swap')
            return setTimeout(() => {
                this.init();
                this.play(0)
            }, 5000)
        }

        var dur = 10;
        if (!this.tog)
            dur = this.snapshots[idx][2] ? 150 : 50


        setTimeout(function () {
                this.relabel()
                const ss = this.snapshots[idx];
                const x = ss[0], y = ss[1], doSwap = ss[2];
                const cls = doSwap ? 'swap' : 'highlight';
                const a = this.dom.find(`[pos=${x}]`), b = this.dom.find(`[pos=${y}]`)

                this.tog = !this.tog
                if (!this.tog) {
                    a.addClass(cls); b.addClass(cls)
                    return this.play(idx)
                } else  {
                    a.removeClass('highlight swap'); b.removeClass('highlight swap')
                    if (this.snapshots[idx][2]) {
                        let tmp = a.css('left')
                        a.css({left: b.css('left')})
                        b.css({left: tmp})
                        return this.play(idx + 1)
                    }
                }
                return this.play(idx + 1)
            }.bind(this), dur)
    }


    selectionSort() {
        const lst = this.lst;
        for (let i = 0; i < lst.length; i++) {
            let minIdx = i
            for (let j = i; j < lst.length; j++) {
                this.swap(j, j, false)
                if (lst[minIdx] > lst[j])
                    minIdx = j
            }
            if (i !== minIdx)
                this.swap(i, minIdx, true)
        }
        return this
    }


    bubbleSort() {
        const lst = this.lst;
        for (let i = lst.length - 1; i > -1; i--)
            for (let j = 0; j < i; j++) {
                this.swap(j, j + 1, lst[j] > lst[j + 1])
            }
        return this
    }

    insertionSort() {
        const lst = this.lst;
        for (let i = 0; i < lst.length; i++) {
            for(var j = i; j > 0 && lst[j - 1] > lst[j]; j--) {
                this.swap(j - 1, j, true)
            }
            if (j > 1)
                this.swap(j - 1, j, false)
        }
        return this
    }

    quickSort(i=0, j=this.lst.length) {
        if (i >= j) return

        let pivot = this.lst[i]
        let a = i + 1
        let b = j - 1
        while (a <= b) {
            while (a <= b && this.lst[a] < pivot)
                this.swap(a++, b, false)

            if (a < b)
                this.swap(a, b, true)
            b--
        }


        this.swap(i, a - 1, true)
        this.quickSort(i, a - 1)
        this.quickSort(a, j)
        return this
    }

    // given the left and right sub-trees are already heaps,
    // make the current tree a heap as well
    bubbleDown(len, i = 0) {
        const getL = (idx) => idx * 2 + 1;
        const getR = (idx) => idx * 2 + 2;
        var lst = this.lst

        let largest = i

        if (getL(i) < len)
            this.swap(getL(i), largest, false)

        if (getR(i) < len)
            this.swap(getR(i), largest, false)

        // check left child
        if (getL(i) < len && lst[largest] < lst[getL(i)])
            largest = getL(i)

        // check right child
        if (getR(i) < len && lst[largest] < lst[getR(i)])
            largest = getR(i)

        // if current node is already largest, no-op is needed
        if (largest === i)
            return

        // otherwise swap with the largest child
        // and enforce the heap property down the subtree with largest value
        this.swap(i, largest, true)
        this.bubbleDown(len, largest)
    }

    heapSort() {
        var lst = this.lst

        // heapify
        for (let i = Math.floor(lst.length / 2) - 1; i > -1; i--) {
            this.bubbleDown(lst.length, i)
        }

        // iteratively pop max from the heap to build sorted list
        for (let i = lst.length - 1; i > -1; i--) {
            this.swap(0, i, true)
            this.bubbleDown(i)
        }
        return this
    }

    shellSort() {
        var lst = this.lst
        for (let gap = lst.length - 1; gap > 0; gap = Math.floor(gap / 2)) {
            for (let end = gap; end < lst.length; end++) {
                for (var i = end - gap, j = end; i >= 0 && lst[i] > lst[j]; i -= gap, j -= gap) {
                    this.swap(i, j)
                }
                if (i > 0)
                    this.swap(i - 1, j - 1, false)
            }
        }
        return this
    }

    combSort() {
        let lst = this.lst
        let getGap = (gap) => Math.floor(gap / 1.3)
        for (let gap = getGap(lst.length); gap > 0; gap = getGap(gap))
            for (let s = 0, e = gap; e < lst.length; s++, e++)
                this.swap(s, e, lst[s] > lst[e])
        return this
    }
}

$(document).ready(function () {
    new Algo($('.sort-algo.selection')).selectionSort().play();
    new Algo($('.sort-algo.bubble')).bubbleSort().play();
    new Algo($('.sort-algo.insertion')).insertionSort().play();
    new Algo($('.sort-algo.quick')).quickSort().play();
    new Algo($('.sort-algo.heap')).heapSort().play();
    new Algo($('.sort-algo.shell')).shellSort().play();
    new Algo($('.sort-algo.comb')).combSort().play();
})


