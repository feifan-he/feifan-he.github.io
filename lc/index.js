let questions, solutions, filtered_questions;
let page_count = 20;
let page_num = 1;

function load_questions(callback) {
    if (localStorage.getItem("password") === null) {
        window.location.href = "/lc/login";
    }

    $.get('questions.txt', function (data) {
        questions = JSON.parse(CryptoJS.AES.decrypt(data, localStorage.getItem('password')).toString(CryptoJS.enc.Utf8));
        questions.splice(0, 0, null);
        if (callback)
            callback();
    }.bind(this));
}

function filter_questions() {
    let companies = $(".companies option:selected").map(function () {
        return this.value
    }).get();
    let tags = $(".tags option:selected").map(function () {
        return this.value
    }).get();
    let difficulty = $(".difficulty option:selected").map(function () {
        return this.value
    }).get();

    page_num = 0
    filtered_questions = questions.filter(function (question) {
        if (question === null) return;
        return question.categoryTitle === 'Algorithms' &&
            (question.companies.filter(value => companies.includes(value)).length > 0 || companies.length === 0) &&
            (question.tags.filter(value => tags.includes(value)).length > 0 || tags.length === 0) &&
            (difficulty.includes(question.difficulty) || difficulty.length === 0)
    });
}

function set_page() {
    let output = [];
    if (page_count * page_num >= filtered_questions.length) {
        page_num = Math.ceil(filtered_questions.length / page_count);
    } else if (page_num < 1) {
        page_num = 1;
    }
    for (q of filtered_questions.slice(page_count * (page_num - 1), page_count * page_num))
        output.push(`<div class="title">\n\n
## ${q.id}. ${q.title}<span class="${q.difficulty.toLowerCase()}"></span></div>\n
<a class="btn btn-outline-dark btn-sm mr-2" href="https://leetcode.com/problems/${q.url}" target="_blank">Leetcode</a>
<button class="btn btn-outline-warning btn-sm mr-2" data-toggle="modal" data-target="#modal" onclick="openCategories('${q.id}')">Categories</button>
<button class="btn btn-outline-info btn-sm" style="${!q.hasSolution ? 'display:none' : ''}" onclick="copySolutions('${q.id}')">Solution</button>
<hr>\n${q.question}`)

    let parsed = marked.parse(output.map(q => `<div class="question-wrapper"> ${q} </div>`).join(''));
    $('.page').val(page_num)
    $('#content').html(parsed);
    $("html, body").animate({scrollTop: 0});
}

function go_to_page(page_inp) {
    page_num = parseInt(page_inp.value);
    set_page();
}

function go_next_page() {
    page_num += 1;
    set_page();
}

function go_prev_page() {
    page_num -= 1;
    set_page()
}

function set_event_listeners() {
    $('.page').keyup(function (event) {
        if (event.originalEvent.key === 'Enter') {
            this.blur();
        }
    })

    $('body').keyup(function (event) {
        switch (event.originalEvent.key) {
            case 'ArrowLeft':
                go_prev_page();
                break;
            case 'ArrowRight':
                go_next_page();
                break;
        }
    })

    $('.next-page').click(go_next_page);
    $('.prev-page').click(go_prev_page);
}

function init_selection_picker() {
    $('select.companies').append("Apple|Facebook|Google|Amazon|Microsoft|Netflix|Airbnb|Uber|Lyft|LinkedIn|DoorDash|instacart|Snapchat|Twitter|ByteDance|tiktok|Dropbox|Robinhood|Paypal|Goldman Sachs|Citadel|Two Sigma|Databricks|Accenture|Accolite|Activision|Adobe|Affirm|Alation|American Express|Arcesium|Arista Networks|Asana|Athenahealth|Atlassian|Audible|Barclays|BlackRock|Bloomberg|Bolt|Booking.com|Box|C3 IoT|Capital One|Cisco|Cloudera|Cohesity|Coursera|Cruise Automation|Dataminr|DE Shaw|Dell|Docusign|Dunzo|eBay|Epic Systems|Expedia|FactSet|Flipkart|GoDaddy|Grab|Groupon|HBO|Hotstar|HRT|Huawei|IBM|Indeed|Infosys|Intel|Intuit|IXL|JPMorgan|Juspay|Karat|LiveRamp|MakeMyTrip|Mathworks|MindTickle|Morgan Stanley|Myntra|Nagarro|National Instruments|Nutanix|Nvidia|Opendoor|Oracle|Palantir Technologies|PayTM|payu|PhonePe|Pinterest|Pure Storage|Qualcomm|Qualtrics|Quora|Reddit|Redfin|Roblox|Rubrik|Salesforce|Samsung|SAP|Sapient|ServiceNow|Shopee|Snapdeal|Splunk|Spotify|Sprinklr|Square|Sumologic|Swiggy|tcs|Tesla|TuSimple|Twilio|Twitch|Visa|VMware|Walmart Global Tech|Wayfair|Yahoo|Yandex|Yelp|Zillow|Zoho|Zomato|Zoom"
        .split('|').map(x => $('<option>').html(x))).selectpicker({actionsBox: true});

    $('select.tags').append(
        "Array|Hash Table|Linked List|Math|Recursion|String|Sliding Window|Binary Search|Divide and Conquer|Dynamic Programming|Two Pointers|Greedy|Sorting|Backtracking|Stack|Heap (Priority Queue)|Merge Sort|String Matching|Bit Manipulation|Matrix|Monotonic Stack|Simulation|Combinatorics|Memoization|Tree|Depth-First Search|Binary Tree|Binary Search Tree|Breadth-First Search|Union Find|Graph|Trie|Design|Doubly-Linked List|Geometry|Interactive|Bucket Sort|Radix Sort|Counting|Data Stream|Iterator|Database|Rolling Hash|Hash Function|Shell|Enumeration|Number Theory|Topological Sort|Prefix Sum|Quickselect|Binary Indexed Tree|Segment Tree|Line Sweep|Ordered Set|Queue|Monotonic Queue|Counting Sort|Brainteaser|Game Theory|Eulerian Circuit|Randomized|Reservoir Sampling|Shortest Path|Bitmask|Rejection Sampling|Probability and Statistics|Suffix Array|Concurrency|Minimum Spanning Tree|Biconnected Component|Strongly Connected Component"
            .split('|').map(x => $('<option>').html(x))).selectpicker({actionsBox: true});

    $('select.difficulty').selectpicker();
}

function openCategories(id) {
    let modal = $('#modal')
    modal.find('.modal-title').html('Categories');
    modal.find('.modal-body').html(questions[id].tags.map(tag => `<div class="badge badge-info tag">${tag}</div>`));
}

function copyToClipboard(str) {
    const el = document.createElement('textarea');
    el.value = str;
    document.body.appendChild(el);
    el.select();
    document.execCommand('copy');
    document.body.removeChild(el);
}

function copySolutions(id) {
    $.get(`questions/${id}.txt`, function (data) {
        let question = JSON.stringify(JSON.parse(CryptoJS.AES.decrypt(data, localStorage.getItem('password')).toString(CryptoJS.enc.Utf8)), null, '  ');
        let to_be_copied = (`
            function main() {
                (function () {
                    // Load the script
                    const script = document.createElement("script");
                    script.src = 'https://ajax.googleapis.com/ajax/libs/jquery/3.6.0/jquery.min.js';
                    script.type = 'text/javascript';
                    script.addEventListener('load', () => {
            
                        let css = {
                            border: '1px solid black',
                            margin: '50px auto',
                            height: 'auto'
                        };
            
                        let app = $('#app');
                        app.css({
                            width: 1000,
                            margin: '50px auto',
                            height: 'auto'
                        }).html('')
                            .append($('<div>').css(css).html(question.questionHTML))
                      .append($('<div>').css(css).html(question.solutionHTML))
                        $('body').css({'overflow': 'scroll'})
                        $('.nav__1n5p, .css-isal7m, .header__28Cb').remove()
                      $('.content__QRGW').css({position: 'relative'})
            
                        console.log(question.documents[0].replace(/:.*/, ''));
            
                      function load_images(container) {
                        return (data) => {
                          console.log(data)
                          for (img of data.timeline)
                            container.append($('<img>').attr('src', img.image).css(
                              {
                                margin: '20px auto',
                                border: '1px solid grey',
                                display: 'block',
                                'max-width': 700,
                                'max-height': 350,
                                height: 'auto'
                              }
                            ))
                        }
                      }
            
                      let containers = $('.dia-container__jsK9')
                      for (let i in question.documents) {
                        let url = question.documents[i];
                        let container = $(containers[i]).html('').css({height: 'inherit', width: 'inherit'});
                        [w, h] = url.split(':')[1].split(',')
                        container.append((container = $('<div>').css({margin: 'auto 0'}))).append($('<hr>'))
                        $.getJSON(url.replace(/:.*/, ''), load_images(container));
                      }
                    });
                    document.head.appendChild(script);
                })();
            }
            
            let question = ${question}
            main();
`.replace(/\n {12}/g, '\n'));
        copyToClipboard(to_be_copied);
    }.bind(this));
}

function search_by_ids(ids) {
    filtered_questions = [];
    for (let id of ids.split(/[,\n; \-*#]/).filter(x => x.trim() !== '')) {
        filtered_questions.push(questions[parseInt(id)]);
    }
    if (filtered_questions.length === 0)
        filter_questions();
    set_page();
}

$(function () {
    load_questions(() => {
        filter_questions();
        set_page();
    });
    init_selection_picker();
    set_event_listeners();
})
