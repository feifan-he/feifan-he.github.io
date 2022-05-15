let questions;
let filtered_questions;
let page_count = 20;
let page_num = 1;

function load_questions(callback) {
    if (localStorage.getItem("password") === null) {
        window.location.href = "/lc/login";
    }

    $.get({
        url: 'questions.txt',
        cache: true
    }, function (data) {
        questions = JSON.parse(CryptoJS.AES.decrypt(data, localStorage.getItem('password')).toString(CryptoJS.enc.Utf8));
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

    filtered_questions = questions.filter(function (question) {
        return (question.companies.filter(value => companies.includes(value)).length > 0 || companies.length === 0) &&
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
        output.push(`<div class="title">\n\n## ${q.id}. ${q.title}<span class="${q.difficulty.toLowerCase()}"></span></div>\n<hr>\n${q.question}`)
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

$(function () {
    $('select.companies').append("Apple|Facebook|Google|Amazon|Microsoft|Netflix|Airbnb|Uber|Lyft|LinkedIn|DoorDash|instacart|Snapchat|Twitter|ByteDance|tiktok|Dropbox|Robinhood|Paypal|Goldman Sachs|Citadel|Two Sigma|Databricks|Accenture|Accolite|Activision|Adobe|Affirm|Alation|American Express|Arcesium|Arista Networks|Asana|Athenahealth|Atlassian|Audible|Barclays|BlackRock|Bloomberg|Bolt|Booking.com|Box|C3 IoT|Capital One|Cisco|Cloudera|Cohesity|Coursera|Cruise Automation|Dataminr|DE Shaw|Dell|Docusign|Dunzo|eBay|Epic Systems|Expedia|FactSet|Flipkart|GoDaddy|Grab|Groupon|HBO|Hotstar|HRT|Huawei|IBM|Indeed|Infosys|Intel|Intuit|IXL|JPMorgan|Juspay|Karat|LiveRamp|MakeMyTrip|Mathworks|MindTickle|Morgan Stanley|Myntra|Nagarro|National Instruments|Nutanix|Nvidia|Opendoor|Oracle|Palantir Technologies|PayTM|payu|PhonePe|Pinterest|Pure Storage|Qualcomm|Qualtrics|Quora|Reddit|Redfin|Roblox|Rubrik|Salesforce|Samsung|SAP|Sapient|ServiceNow|Shopee|Snapdeal|Splunk|Spotify|Sprinklr|Square|Sumologic|Swiggy|tcs|Tesla|TuSimple|Twilio|Twitch|Visa|VMware|Walmart Global Tech|Wayfair|Yahoo|Yandex|Yelp|Zillow|Zoho|Zomato|Zoom"
        .split('|').map(x => $('<option>').html(x))).selectpicker({actionsBox: true});

    $('select.tags').append(
        "Array|Hash Table|Linked List|Math|Recursion|String|Sliding Window|Binary Search|Divide and Conquer|Dynamic Programming|Two Pointers|Greedy|Sorting|Backtracking|Stack|Heap (Priority Queue)|Merge Sort|String Matching|Bit Manipulation|Matrix|Monotonic Stack|Simulation|Combinatorics|Memoization|Tree|Depth-First Search|Binary Tree|Binary Search Tree|Breadth-First Search|Union Find|Graph|Trie|Design|Doubly-Linked List|Geometry|Interactive|Bucket Sort|Radix Sort|Counting|Data Stream|Iterator|Database|Rolling Hash|Hash Function|Shell|Enumeration|Number Theory|Topological Sort|Prefix Sum|Quickselect|Binary Indexed Tree|Segment Tree|Line Sweep|Ordered Set|Queue|Monotonic Queue|Counting Sort|Brainteaser|Game Theory|Eulerian Circuit|Randomized|Reservoir Sampling|Shortest Path|Bitmask|Rejection Sampling|Probability and Statistics|Suffix Array|Concurrency|Minimum Spanning Tree|Biconnected Component|Strongly Connected Component"
            .split('|').map(x => $('<option>').html(x))).selectpicker({actionsBox: true});

    $('select.difficulty').selectpicker();

    load_questions(() => {
        filter_questions()
        set_page()
    })


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


    $('.next-page').click(go_next_page)
    $('.prev-page').click(go_prev_page)
})
