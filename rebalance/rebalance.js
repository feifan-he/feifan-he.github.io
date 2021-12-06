angular.module('rebalanceApp', [])
    .controller('ctl', function() {
        let searchParams = Object.fromEntries(new URLSearchParams(window.location.search).entries())
        this.rows = searchParams.rows ? JSON.parse(searchParams.rows) : [];
        this.addRow = () => {
            let id = this.rows.length === 0 ? 0 : this.rows[this.rows.length - 1].id + 1;
            this.rows.push({symbol: '', weight: 0, current: 0, id: id})
        };
        if (this.rows.length === 0)
            this.addRow();
        this.total = searchParams.total || 100000;
        this.multiplier = searchParams.multiplier || 100;
        this.round = (num) => Math.round(num * 100) / 100
        this.deleteRow = (index) => this.rows.splice(index, 1)
        this.highlight = (cssSelector) => setTimeout(() => $(cssSelector).first().select(), 100)
        this.printMoney = (num) => '$' + num.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",");
        this.doBlur = function($event){

            var target = $event.target;

            // do more here, like blur or other things
            target.blur();
        }

        this.onUnfocus = (keyPress) => {
            if(keyPress.keyCode===13){
                keyPress.target.blur();
            }
        }

        function copyToClipboard(str) {
            const el = document.createElement('textarea');
            el.value = str;
            document.body.appendChild(el);
            el.select();
            document.execCommand('copy');
            document.body.removeChild(el);
        }

        this.copyLink = () => {
            let queryParams = {
                rows: JSON.stringify(this.rows).replace(/\\/g, ""),
                multiplier: this.multiplier,
                total: this.total,
            }
            let suffix = "";
            for (let key in queryParams){
                var val = queryParams[key];
                suffix += `${key}=${val}&`
            }
            suffix = suffix.slice(0, -1);
            copyToClipboard(`${window.location.origin + window.location.pathname}?${suffix}`)
        }
    });