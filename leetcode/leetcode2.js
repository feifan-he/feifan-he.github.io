var script = document.createElement('script');
script.src = 'https://code.jquery.com/jquery-3.4.1.min.js';
script.type = 'text/javascript';
document.getElementsByTagName('head')[0].appendChild(script);

cnt = 100
res = []

function main() {


    let intervalId = setInterval(() => {
        if (!cnt) {
            clearInterval(intervalId);

            var container = $('<div id="question_container">');
            $('#app').css({
                width: 1000,
                'margin': '50px auto',
                'height': 'auto',
                'border': '1px solid black',
            }).html(container);

            container.html($(res.join('<br><hr><br>')))
            $('body').css({'overflow': 'scroll'})
            $('.css-isal7m, .css-q9155n').remove()
        }

        const nextBtn = $('[data-cy="next-question-btn"]');
        if (nextBtn.length === 0)
            return

        const content = $('[data-key="description-content"]');
        if (content.html() === '')
            return

        let textContent = content.html();
        if (!textContent.includes('Subscribe to unlock.')) {
            // container.append(textContent)
            res.push(textContent)
        }


        cnt--;
        nextBtn.click();
    }, 500)
}


setTimeout(main, 1000)