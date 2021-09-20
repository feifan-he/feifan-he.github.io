```python

# ============================================================================

# 1. Two Sum
# Difficulty: Easy
# link: https://leetcode.com/problems/two-sum/
# Companies: Uber,Google,Adobe,Apple,Walmart Labs,Yahoo,LinkedIn,Goldman Sachs,Airbnb,Huawei,Amazon,Facebook,Yandex,Bloomberg,Splunk,Oracle,Cisco,SAP,Microsoft,VMware
# Categories: Array,Hash Table

# ----------------------------------------------------------------------------

class Solution(object):
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        seen = {}
        for i, num in enumerate(nums):
            diff = target - num
            if diff in seen: return [seen[diff], i]
            seen[num] = i


# ============================================================================

# 2. Add Two Numbers
# Difficulty: Medium
# link: https://leetcode.com/problems/add-two-numbers/
# Companies: Uber,Google,Adobe,Apple,Yahoo,Amazon,Facebook,Yandex,Bloomberg,ByteDance,Oracle,Cisco,Microsoft
# Categories: Linked List,Math

# ----------------------------------------------------------------------------

# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def addTwoNumbers(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        l1_cur, l2_cur = l1, l2
        cur = res = ListNode('dummy')
        carry = 0
        while l1 or l2 or carry:
            digit_sum = (l1.val if l1 else 0) + (l2.val if l2 else 0) + carry
            carry, digit = divmod(digit_sum, 10)
            cur.next = ListNode(digit)
            cur = cur.next
            if l1: l1 = l1.next
            if l2: l2 = l2.next
        return res.next


# ============================================================================

# 3. Longest Substring Without Repeating Characters
# Difficulty: Medium
# link: https://leetcode.com/problems/longest-substring-without-repeating-characters/
# Companies: Uber,SAP,Google,Cisco,Alation,Samsung,Atlassian,Goldman Sachs,Zillow,Amazon,Facebook,Bloomberg,ByteDance,Oracle,Adobe,eBay,Microsoft,VMware,Apple
# Categories: Hash Table,Two Pointers,String,Sliding Window

# ----------------------------------------------------------------------------

class Solution(object):
    def lengthOfLongestSubstring(self, s):
        """
        :type s: str
        :rtype: int
        """
        max_len = 0
        visited = {}
        i = 0
        for j, char in enumerate(s):
            if char not in visited or visited[char] < i:
                cur_len = j - i + 1
                max_len = max(max_len, cur_len)
            else:
                i = max(visited[char] + 1, i)

            visited[char] = j

        return max_len


# ============================================================================

# 4. Median of Two Sorted Arrays
# Difficulty: Hard
# link: https://leetcode.com/problems/median-of-two-sorted-arrays/
# Companies: Uber,Google,Adobe,Apple,Zulily,Goldman Sachs,Amazon,Visa,Facebook,Oracle,eBay,Microsoft,Zillow
# Categories: Array,Binary Search,Divide and Conquer

# ----------------------------------------------------------------------------

class Solution(object):
    def findMedianSortedArrays(self, nums1, nums2):
        a, b, c = len(nums1), len(nums2), len(nums1) + len(nums2)
        i = j = k = 0
        prev = cur =  None
        while k < c:
            prev = cur
            if j >= b or ( i < a and nums1[i] <= nums2[j]): cur, i = nums1[i], i + 1
            else: cur, j = nums2[j], j + 1

            if c % 2 and c / 2 == k: return cur
            elif c % 2 == 0 and c / 2 == k: return (cur + prev) / 2.

            k += 1


# ============================================================================

# 5. Longest Palindromic Substring
# Difficulty: Medium
# link: https://leetcode.com/problems/longest-palindromic-substring/
# Companies: Uber,Airbnb,Google,Adobe,Apple,Yahoo,Wayfair,Amazon,Facebook,Bloomberg,Cisco,SAP,Microsoft,ServiceNow
# Categories: String,Dynamic Programming

# ----------------------------------------------------------------------------

class Solution(object):
    def longestPalindrome(self, s):
        """
        :type s: str
        :rtype: str
        """
        def get_pal(i, j):
            while 0 < i and j  < len(s) - 1 and s[i - 1] == s[j + 1]:
                i -= 1
                j += 1
            return [i, j + 1]

        max_pal = ""
        for idx in range(len(s)):
            pal1 = get_pal(idx, idx)
            pal2 = get_pal(idx + 1, idx)
            if pal1[1]-pal1[0] > len(max_pal):
                max_pal = s[pal1[0]: pal1[1]]
            if pal2[1] - pal2[0] > len(max_pal):
                max_pal = s[pal2[0]: pal2[1]]
        return max_pal


# ============================================================================

# 6. ZigZag Conversion
# Difficulty: Medium
# link: https://leetcode.com/problems/zigzag-conversion/
# Companies: Amazon,Adobe
# Categories: String

# ----------------------------------------------------------------------------

class Solution(object):
    def convert(self, s, numRows):
        """
        :type s: str
        :type numRows: int
        :rtype: str"""
        res = [[] for _ in range(numRows)]
        cur, direction = 0, 1
        for i, char in enumerate(s):
            res[cur].append(char)
            direction *= 1 if (0 <= (direction + cur) < numRows) else -1
            cur += direction
        return ''.join([''.join(row) for row in res])


# ============================================================================

# 7. Reverse Integer
# Difficulty: Easy
# link: https://leetcode.com/problems/reverse-integer/
# Companies: Google,Adobe,Apple,Amazon,Facebook,Bloomberg,Microsoft
# Categories: Math

# ----------------------------------------------------------------------------

class Solution(object):
    def reverse(self, x):
        """
        :type x: int
        :rtype: int
        """
        sign = 1 if x >= 0 else -1
        x = abs(x)
        res = 0
        while x:
            res = res * 10 + x % 10
            x /= 10
        return res * sign * (0 if res >> 31 else 1)


# ============================================================================

# 8. String to Integer (atoi)
# Difficulty: Medium
# link: https://leetcode.com/problems/string-to-integer-atoi/
# Companies: Apple,LinkedIn,Goldman Sachs,Amazon,Facebook,Bloomberg,Microsoft
# Categories: Math,String

# ----------------------------------------------------------------------------

class Solution(object):
    def myAtoi(self, s):
        """
        :type str: str
        :rtype: int
        """
        s = s.lstrip(' ')
        sign = 1
        if s.startswith('-'): s, sign = s[1:], -1
        elif s.startswith('+'): s = s[1:]
        s = s[:next((i for i, num in enumerate(s) if not num.isdigit()), len(s))]
        if not s: return 0
        int_rep = reduce(lambda x, y: x * 10 + (ord(y) - ord('0')), s, 0)
        return max(min(int_rep * sign, 2147483647), -2147483648)


# ============================================================================

# 9. Palindrome Number
# Difficulty: Easy
# link: https://leetcode.com/problems/palindrome-number/
# Companies: Amazon,Facebook,Bloomberg
# Categories: Math

# ----------------------------------------------------------------------------

class Solution(object):
    def isPalindrome(self, x):
        """
        :type x: int
        :rtype: bool
        """
        if x < 0: return False
        y, rev = x, 0
        while y:
            rev = rev * 10 + y % 10
            y /= 10
        return rev == x


# ============================================================================

# 10. Regular Expression Matching
# Difficulty: Hard
# link: https://leetcode.com/problems/regular-expression-matching/
# Companies: Uber,Google,Coursera,eBay,Amazon,Facebook,Bloomberg,Microsoft
# Categories: String,Dynamic Programming,Backtracking

# ----------------------------------------------------------------------------

class Solution(object):

    def isMatch(self, s, p):
        A, B = len(s), len(p)
        def does_match(a, b, memo={}):
            if (a, b) in memo: return memo[(a, b)]
            res = False
            if b == B: res = (a == A)
            else:
                is_match_a = a < A and p[b] in [s[a], '.']
                res = does_match(a, b + 2) or is_match_a and does_match(a + 1, b) \
                      if b + 1 < B and p[b + 1] == '*' \
                      else is_match_a and does_match(a + 1, b + 1)
            return memo.setdefault((a, b), res)
        return does_match(0, 0)

    def _isMatch(self, s, p):
        m, n = len(s), len(p)
        dp = [[False] * (m + 1) for _ in range(n + 1)]
        dp[0][0] = True

        for i in range(n):
            dp[i+1][0] = "*" == p[i] and dp[i-1][0]

        is_match = lambda x, y: x in [y, '.']
        for i in range(n):
            for j in range(m):
                if p[i] != '*':
                    dp[i+1][j+1] = is_match(p[i], s[j]) and dp[i][j]
                else:
                    dp[i+1][j+1] = is_match(p[i-1], s[j]) and dp[i+1][j] or dp[i-1][j+1] or dp[i][j+1]
        return dp[-1][-1]


# ============================================================================

# 11. Container With Most Water
# Difficulty: Medium
# link: https://leetcode.com/problems/container-with-most-water/
# Companies: Google,Apple,Goldman Sachs,Amazon,Facebook,ByteDance,Microsoft
# Categories: Array,Two Pointers

# ----------------------------------------------------------------------------

class Solution(object):
    def maxArea(self, height):
        lmax = rmax = 0
        i, j = 0, len(height) - 1
        max_water = 0
        while i < j:
            lmax, rmax = max(lmax, height[i]), max(rmax, height[j])
            max_water = max(max_water, (j - i) * min(height[i], height[j]))
            if lmax < rmax: i += 1
            else: j -= 1
        return max_water


# ============================================================================

# 14. Longest Common Prefix
# Difficulty: Easy
# link: https://leetcode.com/problems/longest-common-prefix/
# Companies: Amazon,Cisco,Twilio
# Categories: String

# ----------------------------------------------------------------------------

class Solution(object):
    def longestCommonPrefix(self, strs):
        if not strs: return ""
        l = min(((s, i) for i, s in enumerate(strs)))[1]
        h = max(((s, i) for i, s in enumerate(strs)))[1]
        return strs[l][:next((i for i in range(len(strs[l])) if strs[l][i] != strs[h][i]), len(strs[l]))]


# ============================================================================

# 15. 3Sum
# Difficulty: Medium
# link: https://leetcode.com/problems/3sum/
# Companies: Uber,Google,Adobe,Apple,Visa,Yahoo,Walmart Labs,Cisco,Amazon,Qualtrics,Facebook,Bloomberg,Square,Oracle,Tencent,Microsoft
# Categories: Array,Two Pointers

# ----------------------------------------------------------------------------

class Solution(object):
    def threeSum(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        nums.sort()
        res = []
        for i in range(len(nums) - 2):
            if i > 0 and nums[i] == nums[i - 1]:
                continue
            i_num = nums[i]
            j = i + 1
            k = len(nums) - 1
            while j < k:
                j_num = nums[j]
                k_num = nums[k]
                total = i_num + j_num + k_num
                if total == 0:
                    res.append([i_num, j_num, k_num])
                    while j < k and nums[j] == nums[j + 1]:
                        j += 1
                    while j < k and nums[k] == nums[k - 1]:
                        k -= 1
                    j += 1
                elif total > 0:
                    k -= 1
                elif total < 0:
                    j += 1
        return res


# ============================================================================

# 16. 3Sum Closest
# Difficulty: Medium
# link: https://leetcode.com/problems/3sum-closest/
# Companies: Google,Bloomberg
# Categories: Array,Two Pointers

# ----------------------------------------------------------------------------

class Solution(object):
    def threeSumClosest(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        min_diff = float('inf')
        min_num = None
        nums.sort()
        for i in range(0, len(nums) - 2):
            j, k = i + 1, len(nums) - 1
            while j < k:
                i_j_k_sum = nums[i] + nums[j] + nums[k]
                diff = abs(i_j_k_sum - target)
                if min_diff > diff:
                    min_diff = diff
                    min_sum = i_j_k_sum
                if i_j_k_sum > target: k -= 1
                else: j += 1
        return min_sum


# ============================================================================

# 17. Letter Combinations of a Phone Number
# Difficulty: Medium
# link: https://leetcode.com/problems/letter-combinations-of-a-phone-number/
# Companies: Uber,Google,Apple,Yahoo,Atlassian,Amazon,Salesforce,Facebook,Bloomberg,Oracle,Dropbox,Microsoft,VMware
# Categories: String,Backtracking

# ----------------------------------------------------------------------------

class Solution(object):
    def letterCombinations(self, digits):
        """
        :type digits: str
        :rtype: List[str]
        """
        if not digits: return []
        mappings = ['', '', 'abc', 'def', 'ghi', 'jkl', 'mno', 'pqrs', 'tuv', 'wxyz']
        res = ['']
        for digit in digits: res = [item + char for item in res for char in mappings[int(digit)]]
        return res


# ============================================================================

# 18. 4Sum
# Difficulty: Medium
# link: https://leetcode.com/problems/4sum/
# Companies: Amazon,Google,Adobe
# Categories: Array,Hash Table,Two Pointers

# ----------------------------------------------------------------------------

class Solution(object):
    def fourSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[List[int]]
        """
        n = len(nums)
        res = set()
        from collections import defaultdict
        sum_to_ind = defaultdict(list)
        for i in range(2, n):
            for j in range(i + 1, n):
                sum_to_ind[(nums[i] + nums[j])].append((i, [nums[i], nums[j]]))
        for i in range(n - 2):
            for j in range(i + 1, n - 1):
                pair_1 = [nums[i], nums[j]]
                pair_1_sum = sum(pair_1)
                new_tar = target - pair_1_sum
                if new_tar in sum_to_ind:
                    for idx, pair_2 in reversed(sum_to_ind[new_tar]):
                        if idx <= j: break
                        res.add(tuple(sorted(pair_1 + pair_2)))
        return list(res)


# ============================================================================

# 19. Remove Nth Node From End of List
# Difficulty: Medium
# link: https://leetcode.com/problems/remove-nth-node-from-end-of-list/
# Companies: Oracle,Yandex,Microsoft,Apple
# Categories: Linked List,Two Pointers

# ----------------------------------------------------------------------------

# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def removeNthFromEnd(self, head, n):
        """
        :type head: ListNode
        :type n: int
        :rtype: ListNode
        """
        nth = head
        for _ in range(n): nth = nth.next
        nth, cur, prev = head, nth, None
        while cur: cur, prev, nth = cur.next, nth, nth.next
        if nth == head: return head.next
        if prev and prev.next: prev.next = prev.next.next
        return head


# ============================================================================

# 20. Valid Parentheses
# Difficulty: Easy
# link: https://leetcode.com/problems/valid-parentheses/
# Companies: Uber,Google,Adobe,Apple,Walmart Labs,IBM,Visa,JPMorgan,Audible,Amazon,Salesforce,Facebook,LinkedIn,Bloomberg,Oracle,Expedia,Citadel,Microsoft,VMware,Yandex
# Categories: String,Stack

# ----------------------------------------------------------------------------

class Solution(object):
    def isValid(self, s):
        """
        :type s: str
        :rtype: bool
        """
        stack = []
        closing_to_opening = {')':'(', '}':'{', ']':'['}
        for char in s:
            if char in '()[]{}':
                if char in closing_to_opening:
                    if not stack or stack[-1] != closing_to_opening[char]:
                        return False
                    stack.pop()
                else:
                    stack.append(char)
        return not stack


# ============================================================================

# 21. Merge Two Sorted Lists
# Difficulty: Easy
# link: https://leetcode.com/problems/merge-two-sorted-lists/
# Companies: Uber,Adobe,Apple,Yahoo,Atlassian,Amazon,Tencent,Yandex,Bloomberg,ByteDance,Facebook,Microsoft,VMware
# Categories: Linked List

# ----------------------------------------------------------------------------

# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def mergeTwoLists(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        dummy = l = ListNode('dummy')
        while l1 or l2:
            if l1 and l2: l1, l2 = (l1, l2) if l1.val < l2.val else (l2, l1)
            else: l1, l2 = l1 or l2, None
            l.next = l1
            l, l1 = l.next, l1.next
        return dummy.next


# ============================================================================

# 22. Generate Parentheses
# Difficulty: Medium
# link: https://leetcode.com/problems/generate-parentheses/
# Companies: Uber,Lyft,Apple,Samsung,Amazon,Google,Walmart Labs,Yandex,Bloomberg,Facebook,Microsoft
# Categories: String,Backtracking

# ----------------------------------------------------------------------------

class Solution(object):
    def generateParenthesis(self, n):
        """
        :type n: int
        :rtype: List[str]
        """
        def _generateParenthesis(so_far=[], open_paren=0, close_paren=0, res=[]):
            if n == open_paren == close_paren:res.append(''.join(so_far))
            elif open_paren > n or close_paren > open_paren: pass
            else:
                so_far.append('(')
                _generateParenthesis(so_far, open_paren + 1, close_paren, res)
                so_far.pop()

                so_far.append(')')
                _generateParenthesis(so_far, open_paren, close_paren + 1, res)
                so_far.pop()

            return res
        return _generateParenthesis()


# ============================================================================

# 23. Merge k Sorted Lists
# Difficulty: Hard
# link: https://leetcode.com/problems/merge-k-sorted-lists/
# Companies: Uber,Google,Apple,Bloomberg,Wish,Twitter,Yahoo,LinkedIn,eBay,Atlassian,Amazon,Walmart Labs,Yandex,ByteDance,Oracle,Facebook,Microsoft
# Categories: Linked List,Divide and Conquer,Heap

# ----------------------------------------------------------------------------

# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def mergeKLists(self, lists):
        """
        :type lists: List[ListNode]
        :rtype: ListNode
        """
        import heapq
        heap = [(lst.val, lst) for lst in lists if lst]
        heapq.heapify(heap)
        cur = dummy_head = ListNode('dummy')
        while heap:
            elem, lst = heapq.heappop(heap)
            cur.next = lst
            cur, lst = cur.next, lst.next
            if lst: heapq.heappush(heap, (lst.val, lst))
        return dummy_head.next


# ============================================================================

# 24. Swap Nodes in Pairs
# Difficulty: Medium
# link: https://leetcode.com/problems/swap-nodes-in-pairs/
# Companies: Uber,Amazon,Facebook,Microsoft
# Categories: Linked List

# ----------------------------------------------------------------------------

# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def swapPairs(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        dummy_head = ListNode('dummy')
        dummy_head.next = head
        cur = dummy_head
        while cur and cur.next and cur.next.next:
            nodes = [cur, cur.next, cur.next.next, cur.next.next.next]
            nodes[0].next, nodes[1].next, nodes[2].next, cur = nodes[2], nodes[3], nodes[1], nodes[1]
        return dummy_head.next


# ============================================================================

# 26. Remove Duplicates from Sorted Array
# Difficulty: Easy
# link: https://leetcode.com/problems/remove-duplicates-from-sorted-array/
# Companies: Google,Facebook,Microsoft
# Categories: Array,Two Pointers

# ----------------------------------------------------------------------------

class Solution(object):
    def removeDuplicates(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        i = 0
        nums.append('dummy')
        for j in xrange(1, len(nums)):
            if nums[j - 1] != nums[j]:
                nums[i] = nums[j - 1]
                i += 1
        while i != len(nums): nums.pop()


# ============================================================================

# 27. Remove Element
# Difficulty: Easy
# link: https://leetcode.com/problems/remove-element/
# Companies: Microsoft
# Categories: Array,Two Pointers

# ----------------------------------------------------------------------------

class Solution(object):
    def removeElement(self, nums, val):
        """
        :type nums: List[int]
        :type val: int
        :rtype: int
        """
        i = 0
        for j, num in enumerate(nums):
            if val != num:
                nums[i] = nums[j]
                i += 1
        while i != len(nums): nums.pop()
        return i


# ============================================================================

# 28. Implement strStr()
# Difficulty: Easy
# link: https://leetcode.com/problems/implement-strstr/
# Companies: Google,Amazon,Facebook,Bloomberg,Dropbox,Microsoft
# Categories: Two Pointers,String

# ----------------------------------------------------------------------------

class Solution(object):
    def strStr(self, haystack, needle):
        """
        :type haystack: str
        :type needle: str
        :rtype: int
        """
        for i in range(0, len(haystack) - len(needle) + 1):
            if haystack[i: i + len(needle)] == needle:
                return i
        return -1


# ============================================================================

# 30. Substring with Concatenation of All Words
# Difficulty: Hard
# link: https://leetcode.com/problems/substring-with-concatenation-of-all-words/
# Companies: Amazon,Google,Microsoft
# Categories: Hash Table,Two Pointers,String

# ----------------------------------------------------------------------------

class Solution(object):
    def findSubstring(self, s, words):
        res = []
        from collections import deque, Counter
        if not words or not s: return []
        word_l, l_words = len(words[0]), len(words)
        cnt_words = Counter(words)
        cnts = [Counter() for _ in range(word_l)]
        for i in range(word_l - 1, len(s)):
            cnt = cnts[i%word_l]
            cnt[s[i - word_l + 1: i + 1]] += 1
            if ((i + 1) / word_l) > l_words:
                word_to_remove = s[i - word_l*(l_words+1) + 1: i - word_l*l_words + 1]
                cnt[word_to_remove] -= 1
                if not cnt[word_to_remove]: del cnt[word_to_remove]
            if ((i + 1) / word_l) >= l_words and all(cnt_words[word] == cnt[word] for word in cnt_words):
                res.append(i - word_l * l_words + 1)
        return res


# ============================================================================

# 31. Next Permutation
# Difficulty: Medium
# link: https://leetcode.com/problems/next-permutation/
# Companies: Uber,Google,Apple,Bloomberg,Amazon,Facebook,ByteDance,Microsoft
# Categories: Array

# ----------------------------------------------------------------------------

class Solution(object):
    def nextPermutation(self, nums):
        if len(nums) <= 1: return
        l, r = 0, len(nums) - 1
        # find first decreasing pair from the right,
        # then swap it with the smallest element that's strictly greater.
        for i in xrange(len(nums) - 2, -1, -1):
            if nums[i] < nums[i+1]:
                right_greater = min(((nums[j], j) for j in range(i + 1, len(nums)) if nums[j] > nums[i]), \
                                    key=lambda x: (x[0], -x[1]))[1]
                nums[i], nums[right_greater] = nums[right_greater], nums[i]
                l = i + 1
                break
        # reverse the elems to the right
        while l < r:
            nums[l], nums[r] = nums[r], nums[l]
            l, r = l + 1, r - 1


# ============================================================================

# 33. Search in Rotated Sorted Array
# Difficulty: Medium
# link: https://leetcode.com/problems/search-in-rotated-sorted-array/
# Companies: Nutanix,Google,Adobe,ByteDance,Samsung,eBay,Amazon,Expedia,Facebook,Bloomberg,Oracle,Walmart Labs,Twitch,Microsoft,Apple
# Categories: Array,Binary Search

# ----------------------------------------------------------------------------

class Solution(object):
    def search(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        lo, hi = 0, len(nums) - 1
        while lo <= hi:
            mid = (lo + hi) / 2
            if nums[mid] == target: return mid
            elif nums[lo] <= target < nums[mid] or \
                (nums[lo] > nums[mid] and (target < nums[mid] or target >= nums[lo])): hi = mid - 1
            else: lo = mid + 1
        return -1


# ============================================================================

# 34. Find First and Last Position of Element in Sorted Array
# Difficulty: Medium
# link: https://leetcode.com/problems/find-first-and-last-position-of-element-in-sorted-array/
# Companies: Uber,Google,Apple,LinkedIn,Amazon,Facebook,Yandex,Oracle
# Categories: Array,Binary Search

# ----------------------------------------------------------------------------

class Solution(object):
    def searchRange(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        res = [-1, -1]
        def _search(i=0, j=len(nums)):
            if i < j:
                mid = (i + j) / 2
                if nums[mid] == target:
                    if mid == 0 or nums[mid - 1] != target: res[0] = mid
                    else: _search(i, mid)

                    if mid == len(nums) - 1 or nums[mid + 1] != target: res[1] = mid
                    else: _search(mid + 1, j)
                else:
                    if nums[i] <= target < nums[mid]: _search(i, mid)
                    elif nums[mid] < target <= nums[j - 1]: _search(mid + 1, j)
            return res
        return _search()


# ============================================================================

# 35. Search Insert Position
# Difficulty: Easy
# link: https://leetcode.com/problems/search-insert-position/
# Companies: Google,Bloomberg
# Categories: Array,Binary Search

# ----------------------------------------------------------------------------

class Solution(object):
    def searchInsert(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        i, j = 0, len(nums)
        while i < j:
            mid = (i + j) / 2
            if nums[mid] == target: return mid
            elif nums[mid] > target: j = mid
            else: i = mid + 1
        return i


# ============================================================================

# 36. Valid Sudoku
# Difficulty: Medium
# link: https://leetcode.com/problems/valid-sudoku/
# Companies: Uber,Amazon,Google,Facebook,Microsoft
# Categories: Hash Table

# ----------------------------------------------------------------------------

class Solution(object):
    def isValidSudoku(self, board):
        visited = set()
        return all(x not in visited and (not visited.add(x))
                   for i, row in enumerate(board) for j, el in enumerate(row)
                   if el != '.' for x in [(i/3, j/3, el), (i, el), ('#', j, el)])


# ============================================================================

# 37. Sudoku Solver
# Difficulty: Hard
# link: https://leetcode.com/problems/sudoku-solver/
# Companies: Amazon,Facebook
# Categories: Hash Table,Backtracking

# ----------------------------------------------------------------------------

class Solution(object):
    def solveSudoku(self, board):

        seen = set()

        def is_valid_add(i, j, el):
            seen_item = {(i, None, el), (None, j, el), (i/3, j/3, el)}
            if seen_item & seen: return False
            board[i][j] = el
            seen.update(seen_item)
            return True

        def remove_seen_item(i, j, el):
            for el in {(i, None, el), (None, j, el), (i/3, j/3, el)}:
                seen.remove(el)
            board[i][j] = '.'

        for i, col in enumerate(board):
            for j, el in enumerate(col):
                if el != '.': is_valid_add(i, j, el)  # add pre-existing

        def _solveSudoku(start_i):
            for i in range(start_i, 9 * 9):
                a, b = divmod(i, 9)
                if board[a][b] == '.':
                    for potential in map(str, range(1, 10)):
                        if is_valid_add(a, b, potential):
                            if _solveSudoku(i + 1): return True
                            remove_seen_item(a, b, potential)
                    return False
            return True
        _solveSudoku(0)


# ============================================================================

# 38. Count and Say
# Difficulty: Easy
# link: https://leetcode.com/problems/count-and-say/
# Companies: Epic Systems,Amazon,Google,Facebook,Microsoft
# Categories: String

# ----------------------------------------------------------------------------

class Solution(object):
    def countAndSay(self, n):
        """
        :type n: int
        :rtype: str
        """
        cur = ['1']
        for i in range(n - 1):
            stack = []
            for elem in cur:
                if not stack or stack[-1][1] != elem: stack.append([1, elem])
                else: stack[-1][0] += 1
            cur = [str(i) for pair in stack for i in pair]
        return ''.join(cur)


# ============================================================================

# 39. Combination Sum
# Difficulty: Medium
# link: https://leetcode.com/problems/combination-sum/
# Companies: Google,Airbnb,Amazon,Square,Facebook,Microsoft
# Categories: Array,Backtracking

# ----------------------------------------------------------------------------

class Solution(object):
    def combinationSum(self, candidates, target):
        from collections import Counter
        def _comb_sum(i, tar, cur=Counter()):
            if tar == 0: self.res.append([y for x in [[coin] * cnt for coin, cnt in cur.iteritems()] for y in x])
            elif tar < 0 or i >= len(candidates): return
            else:
                cand = candidates[i]
                for cnt in range(tar / cand + 1):
                    cur[cand] += cnt
                    _comb_sum(i + 1, tar - cnt  * cand)
                    cur[cand] -= cnt
        self.res = []
        _comb_sum(0, target)
        return self.res


# ============================================================================

# 40. Combination Sum II
# Difficulty: Medium
# link: https://leetcode.com/problems/combination-sum-ii/
# Companies: Uber,Amazon,Microsoft,Apple
# Categories: Array,Backtracking

# ----------------------------------------------------------------------------

class Solution(object):
    def combinationSum2(self, candidates, target):
        from collections import Counter
        coin_cnt = Counter(candidates)
        distinct = coin_cnt.keys()
        def _comb_sum(i, tar, cur=Counter()):
            if tar == 0: self.res.append([y for x in [[coin] * cnt for coin, cnt in cur.iteritems()] for y in x])
            elif tar < 0 or i >= len(distinct): return
            else:
                cand = distinct[i]
                for cnt in range(coin_cnt[cand] + 1):
                    cur[cand] += cnt
                    _comb_sum(i + 1, tar - cnt  * cand)
                    cur[cand] -= cnt

        self.res = []
        _comb_sum(0, target)
        return self.res


# ============================================================================

# 41. First Missing Positive
# Difficulty: Hard
# link: https://leetcode.com/problems/first-missing-positive/
# Companies: Google,Wish,Databricks,Amazon,Facebook,ByteDance,Walmart Labs,Microsoft
# Categories: Array

# ----------------------------------------------------------------------------

class Solution(object):
    def firstMissingPositive(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        for i, num in enumerate(nums):
            if num <= 0: nums[i] = len(nums) + 1
        nums.append(len(nums) + 1)
        for i, num in enumerate(nums):
            if 0 < abs(num) < len(nums): nums[abs(num)] = -abs(nums[abs(num)])
        return next((i for i, num in enumerate(nums) if i and num > 0), len(nums))


# ============================================================================

# 42. Trapping Rain Water
# Difficulty: Hard
# link: https://leetcode.com/problems/trapping-rain-water/
# Companies: Uber,Google,Adobe,Apple,Wish,Databricks,Tableau,Goldman Sachs,Amazon,Qualtrics,Facebook,Bloomberg,Visa,Walmart Labs,Citadel,Microsoft,Flipkart
# Categories: Array,Two Pointers,Stack

# ----------------------------------------------------------------------------

class Solution(object):
    def trap(self, height):
        """
        :type height: List[int]
        :rtype: int
        """
        if len(height) <= 2: return 0
        l, r = 0, len(height) - 1
        res = 0
        l_max = r_max = 0
        while l < r:
            l_max, r_max = max(l_max, height[l]), max(r_max, height[r])
            if l_max < r_max:
                l += 1
                res += max(min(l_max, r_max) - height[l], 0)
            else:
                r -= 1
                res += max(min(l_max, r_max) - height[r], 0)
        return res


# ============================================================================

# 45. Jump Game II
# Difficulty: Hard
# link: https://leetcode.com/problems/jump-game-ii/
# Companies: Nutanix,Amazon,Facebook
# Categories: Array,Greedy

# ----------------------------------------------------------------------------

class Solution(object):

    #for loop
    def jump(self, nums):
        s = e = res = 0
        for i, n in enumerate(nums):
            if i > s:
                s = e = e
                res += 1
            e = max(e, i + n)
        return res

    # while loop
    def _jump(self, nums):
        s = e = i = res = 0
        while e < len(nums) - 1:
            while e < len(nums) - 1 and i <= s:
                e = max(e, nums[i] + i)
                i += 1
            s = e = e
            res += 1
        return res


# ============================================================================

# 46. Permutations
# Difficulty: Medium
# link: https://leetcode.com/problems/permutations/
# Companies: Google,Adobe,Apple,Atlassian,LinkedIn,Amazon,Facebook,Bloomberg,Microsoft
# Categories: Backtracking

# ----------------------------------------------------------------------------

class Solution(object):
    def permute(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        res = [[]]
        for num in nums:
            new_res =[]
            for item in res:
                for i in range(len(item) + 1):
                    new_res.append(item[:i] + [num] + item[i:])
            res = new_res
        return res


# ============================================================================

# 47. Permutations II
# Difficulty: Medium
# link: https://leetcode.com/problems/permutations-ii/
# Companies: Amazon,LinkedIn
# Categories: Backtracking

# ----------------------------------------------------------------------------

class Solution(object):
    def permuteUnique(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        from collections import Counter
        cnt = Counter(nums)

        def _gen_permutation(cur=[], res=set()):
            if not cnt:
                res.add(tuple(cur[:]))
                return res
            for key in cnt.keys():
                cnt[key] -= 1
                cur.append(key)
                if not cnt[key]:
                    del cnt[key]
                _gen_permutation(cur, res)
                cur.pop()
                cnt[key] += 1
            return res
        return _gen_permutation()


# ============================================================================

# 48. Rotate Image
# Difficulty: Medium
# link: https://leetcode.com/problems/rotate-image/
# Companies: Lyft,Cisco,Groupon,Amazon,Microsoft,Palantir Technologies
# Categories: Array

# ----------------------------------------------------------------------------

class Solution(object):
    def rotate(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: void Do not return anything, modify matrix in-place instead.
        """
        for i in range(len(matrix) / 2):
            for j in range(i, len(matrix) - i - 1):
                n = len(matrix) - 1
                vals = matrix[i][j], matrix[j][n - i], matrix[n - i][n - j], matrix[n - j][i]
                matrix[i][j], matrix[j][n - i], matrix[n - i][n - j], matrix[n - j][i] = vals[3], vals[0], vals[1], vals[2]


# ============================================================================

# 49. Group Anagrams
# Difficulty: Medium
# link: https://leetcode.com/problems/group-anagrams/
# Companies: Uber,Google,Adobe,Apple,Affirm,Docusign,Wish,Booking.com,Yahoo,Goldman Sachs,Hulu,Amazon,Facebook,Yandex,Bloomberg,Walmart Labs,Microsoft
# Categories: Hash Table,String

# ----------------------------------------------------------------------------

class Solution(object):
    def groupAnagrams(self, strs):
        """
        :type strs: List[str]
        :rtype: List[List[str]]
        """
        group = {}
        for word in strs:
            hashed_bucket = [0] * 26
            for char in word:
                hashed_bucket[ord(char) % len(hashed_bucket)] += 1
            key = tuple(hashed_bucket)
            group.setdefault(key, [])
            group[key].append(word)
        return group.values()


# ============================================================================

# 51. N-Queens
# Difficulty: Hard
# link: https://leetcode.com/problems/n-queens/
# Companies: Amazon,Facebook,Microsoft,Rubrik
# Categories: Backtracking

# ----------------------------------------------------------------------------

class Solution(object):
    def solveNQueens(self, n):
        """
        :type n: int
        :rtype: List[List[str]]
        """
        vertical, diag1, diag2 = [[False] * (2 * n) for _ in range(3)]
        cur = [['.'] * n for _ in range(n)]
        res = []
        def _solveNQueens(y):
            if y >= n: return
            for x in range(0, n):
                diag1_i, diag2_i = (x - y) % (2 * n), x + y
                if not vertical[x] and not diag1[diag1_i] and not diag2[diag2_i]:
                    cur[x][y] = 'Q'
                    vertical[x] = diag1[diag1_i] = diag2[diag2_i] = True
                    if y == n - 1: res.append([''.join(row) for row in cur])
                    else: _solveNQueens(y + 1)
                    vertical[x] = diag1[diag1_i] = diag2[diag2_i] = False
                    cur[x][y] = '.'
        _solveNQueens(0)
        return res


# ============================================================================

# 52. N-Queens II
# Difficulty: Hard
# link: https://leetcode.com/problems/n-queens-ii/
# Companies: Zenefits
# Categories: Backtracking

# ----------------------------------------------------------------------------

class Solution(object):
    def totalNQueens(self, n):
        """
        :type n: int
        :rtype: List[List[str]]
        """
        vertical, diag1, diag2 = [[False] * (2 * n) for _ in range(3)]
        self.count = 0
        def _totalNQueens(y):
            if y >= n: return
            for x in range(0, n):
                diag1_i, diag2_i = (x - y) % (2 * n), x + y
                if not vertical[x] and not diag1[diag1_i] and not diag2[diag2_i]:
                    vertical[x] = diag1[diag1_i] = diag2[diag2_i] = True
                    if y == n - 1: self.count += 1
                    else: _totalNQueens(y + 1)
                    vertical[x] = diag1[diag1_i] = diag2[diag2_i] = False
        _totalNQueens(0)
        return self.count


# ============================================================================

# 53. Maximum Subarray
# Difficulty: Easy
# link: https://leetcode.com/problems/maximum-subarray/
# Companies: Google,Adobe,Apple,Paypal,Atlassian,LinkedIn,Alibaba,Amazon,Facebook,Bloomberg,ByteDance,Oracle,Walmart Labs,eBay,Microsoft
# Categories: Array,Divide and Conquer,Dynamic Programming

# ----------------------------------------------------------------------------

class Solution(object):
    def maxSubArray(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        res = max_so_far = float('-inf')
        for i, n in enumerate(nums):
            max_so_far = max(max_so_far + n, n)
            res = max(max_so_far, res)
        return res


# ============================================================================

# 54. Spiral Matrix
# Difficulty: Medium
# link: https://leetcode.com/problems/spiral-matrix/
# Companies: Uber,Google,Apple,Robinhood,Paypal,Snapchat,Goldman Sachs,Amazon,Facebook,eBay,Microsoft
# Categories: Array

# ----------------------------------------------------------------------------

class Solution(object):
    def spiralOrder(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: List[int]
        """
        if not matrix: return matrix
        move = [
            lambda (x, y): (x, y + 1),
            lambda (x, y): (x + 1, y),
            lambda (x, y): (x, y - 1),
            lambda (x, y): (x - 1, y),
        ]
        cur_dir = 0
        res = []
        visited = set()
        m, n = len(matrix), len(matrix[0])
        pos = (0, -1)
        for i in range(m * n):
            new_x, new_y = move[cur_dir % 4](pos)
            if not (0 <= new_x < m and 0 <= new_y < n) or \
                    (new_x, new_y) in visited:
                cur_dir += 1
            pos = move[cur_dir % 4](pos)
            res.append(matrix[pos[0]][pos[1]])
            visited.add(pos)
        return res


# ============================================================================

# 55. Jump Game
# Difficulty: Medium
# link: https://leetcode.com/problems/jump-game/
# Companies: Amazon,Google
# Categories: Array,Greedy

# ----------------------------------------------------------------------------

class Solution(object):
    def canJump(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        max_idx_jump = 0
        for i in range(len(nums)):
            if max_idx_jump < i:
                return False
            max_idx_jump = max(max_idx_jump, nums[i] + i)
        return True


# ============================================================================

# 56. Merge Intervals
# Difficulty: Medium
# link: https://leetcode.com/problems/merge-intervals/
# Companies: Uber,SAP,Google,Cisco,Wish,Cruise Automation,LinkedIn,Postmates,VMware,Amazon,Salesforce,Facebook,Snapchat,Bloomberg,Oracle,Adobe,Alibaba,Zulily,Palantir Technologies,Sumologic
# Categories: Array,Sort

# ----------------------------------------------------------------------------

# Definition for an interval.
# class Interval(object):
#     def __init__(self, s=0, e=0):
#         self.start = s
#         self.end = e

class Solution(object):
    def merge(self, intervals):
        """
        :type intervals: List[Interval]
        :rtype: List[Interval]
        """
        res = []
        intervals = [[interval.start, interval.end] for interval in intervals]
        intervals.sort(reverse=True)
        while intervals:
            res.append(intervals.pop())
            if len(res) >= 2:
                [a,b], [c,d] = res[-2], res[-1]
                if a <= c <= b:
                    res.pop(); res.pop()
                    res.append([a, max(b, d)])
        return [Interval(start, end) for start, end in res]


# ============================================================================

# 57. Insert Interval
# Difficulty: Hard
# link: https://leetcode.com/problems/insert-interval/
# Companies: Uber,Google,Twitter,LinkedIn,Amazon,Microsoft
# Categories: Array,Sort

# ----------------------------------------------------------------------------

# Definition for an interval.
# class Interval(object):
#     def __init__(self, s=0, e=0):
#         self.start = s
#         self.end = e

class Solution(object):
    def insert(self, intervals, newInterval):
        """
        :type intervals: List[Interval]
        :type newInterval: Interval
        :rtype: List[Interval]
        """
        left, right = [], []
        s, e = newInterval.start, newInterval.end
        for interval in intervals:
            if interval.end < s: left.append(interval)
            elif e < interval.start: right.append(interval)
            else: s, e = min(interval.start, s), max(interval.end, e)
        return left + [Interval(s, e)] + right


# ============================================================================

# 58. Length of Last Word
# Difficulty: Easy
# link: https://leetcode.com/problems/length-of-last-word/
# Companies:
# Categories: String

# ----------------------------------------------------------------------------

class Solution(object):
    def lengthOfLastWord(self, s):
        """
        :type s: str
        :rtype: int
        """
        s = s.strip(' ')
        return len(s) - s.rfind(' ') - 1


# ============================================================================

# 59. Spiral Matrix II
# Difficulty: Medium
# link: https://leetcode.com/problems/spiral-matrix-ii/
# Companies: Amazon,Adobe,Microsoft,Yandex
# Categories: Array

# ----------------------------------------------------------------------------

class Solution(object):
    def generateMatrix(self, n):
        """
        :type n: int
        :rtype: List[List[int]]
        """
        res = [[None] * n for _ in xrange(n)]
        cur_dir = x = 0
        y = -1
        new_pos_lambda = {
            0: lambda x, y: (x, y + 1),
            1: lambda x, y: (x + 1, y),
            2: lambda x, y: (x, y - 1),
            3: lambda x, y: (x - 1, y)
        }
        for i in xrange(1, n * n + 1):
            new_x, new_y = new_pos_lambda[cur_dir % 4](x, y)
            if not (0 <= new_x < n and  0 <= new_y < n) or res[new_x][new_y] is not None:
                cur_dir += 1
                new_x, new_y = new_pos_lambda[cur_dir % 4](x, y)
            res[new_x][new_y] = i
            x, y = new_x, new_y
        return res


# ============================================================================

# 61. Rotate List
# Difficulty: Medium
# link: https://leetcode.com/problems/rotate-list/
# Companies: Amazon,Microsoft,LinkedIn
# Categories: Linked List,Two Pointers

# ----------------------------------------------------------------------------

# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def rotateRight(self, head, k):
        """
        :type head: ListNode
        :type k: int
        :rtype: ListNode
        """
        if not head:
            return
        count = 0
        cur = head
        tail = None
        while cur:
            count += 1
            tail, cur = cur, cur.next
        k = k % count
        if not k:
            return head
        cur = head
        for i in xrange(count - k - 1):
            cur = cur.next
        head, cur.next, tail.next = cur.next, None, head
        return head


# ============================================================================

# 62. Unique Paths
# Difficulty: Medium
# link: https://leetcode.com/problems/unique-paths/
# Companies: Uber,Mathworks,Amazon,Facebook
# Categories: Array,Dynamic Programming

# ----------------------------------------------------------------------------

class Solution(object):
    def uniquePaths(self, m, n):
        """
        :type m: int
        :type n: int
        :rtype: int
        """
        dp = [[1] * n for _ in range(m)]
        for i in range(1, m):
            for j in range(1, n):
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
        return dp[-1][-1]


# ============================================================================

# 63. Unique Paths II
# Difficulty: Medium
# link: https://leetcode.com/problems/unique-paths-ii/
# Companies: Amazon
# Categories: Array,Dynamic Programming

# ----------------------------------------------------------------------------

class Solution(object):
    def uniquePathsWithObstacles(self, obstacleGrid):
        """
        :type obstacleGrid: List[List[int]]
        :rtype: int
        """
        if not obstacleGrid or not obstacleGrid[0]: return 0
        dp = [[1 - item for item in row] for row in obstacleGrid]
        for i in range(1, len(dp)): dp[i][0] = min(dp[i - 1][0], dp[i][0])
        for j in range(1, len(dp[0])): dp[0][j] = min(dp[0][j - 1], dp[0][j])
        for i in range(1, len(dp)):
            for j in range(1, len(dp[0])):
                if dp[i][j]: dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
        return dp[-1][-1]


# ============================================================================

# 64. Minimum Path Sum
# Difficulty: Medium
# link: https://leetcode.com/problems/minimum-path-sum/
# Companies: Amazon,Apple,Goldman Sachs
# Categories: Array,Dynamic Programming

# ----------------------------------------------------------------------------

class Solution(object):
    def minPathSum(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        for i in range(1, len(grid)): grid[i][0] += grid[i - 1][0]
        for j in range(1, len(grid[0])): grid[0][j] += grid[0][j - 1]
        for i in range(1, len(grid)):
            for j in range(1, len(grid[0])):
                grid[i][j] += min(grid[i-1][j], grid[i][j - 1])
        return grid[-1][-1]


# ============================================================================

# 66. Plus One
# Difficulty: Easy
# link: https://leetcode.com/problems/plus-one/
# Companies: Amazon,Google
# Categories: Array

# ----------------------------------------------------------------------------

class Solution(object):
    def plusOne(self, digits):
        """
        :type digits: List[int]
        :rtype: List[int]
        """
        for i in range(len(digits) - 1, -1, -1):
            if digits[i] < 9:
                digits[i] += 1
                return digits
            else: digits[i] = 0
        digits.insert(0, 1)
        return digits


# ============================================================================

# 67. Add Binary
# Difficulty: Easy
# link: https://leetcode.com/problems/add-binary/
# Companies: Amazon,Google,Facebook
# Categories: Math,String

# ----------------------------------------------------------------------------

class Solution(object):
    def addBinary(self, a, b):
        """
        :type a: str
        :type b: str
        :rtype: str
        """
        a, b, carry = [int(i) for i in list(a)], [int(i) for i in list(b)], 0
        res = []
        while carry or a or b:
            cur = (a.pop() if a else 0) + (b.pop() if b else 0) + carry
            carry, digit = cur / 2, cur % 2
            res.append(str(digit))
        return ''.join(list(reversed(res)))


# ============================================================================

# 68. Text Justification
# Difficulty: Hard
# link: https://leetcode.com/problems/text-justification/
# Companies: Box,Uber,Google,Intuit,LinkedIn,Airbnb,Amazon,Twilio,Microsoft
# Categories: String

# ----------------------------------------------------------------------------

class Solution(object):
    def fullJustify(self, words, maxWidth):
        res = []
        row_char_cnt = 0

        for word in words:
            if row_char_cnt == 0 or row_char_cnt + 1 + len(word) > maxWidth:
                res.append([word])
                row_char_cnt = len(word)
            else:
                res[-1].append(word)
                row_char_cnt += 1 + len(word)

        for i in range(len(res)):
            if i == len(res) - 1 or len(res[i]) == 1:
                res[i] = " ".join(res[i])
                res[i] += " " * (maxWidth - len(res[i]))
            else:
                avg_space, addl_space = divmod(maxWidth - sum(map(len, res[i])), len(res[i]) - 1)
                res[i] = ''.join([res[i][j] + " " * (avg_space + bool(j < addl_space))
                                  for j in range(len(res[i]) - 1)] + [res[i][-1]])
        return res


# ============================================================================

# 70. Climbing Stairs
# Difficulty: Easy
# link: https://leetcode.com/problems/climbing-stairs/
# Companies: Apple,Microsoft,Amazon,Facebook,Bloomberg,Zulily
# Categories: Dynamic Programming

# ----------------------------------------------------------------------------

class Solution(object):
    def climbStairs(self, n):
        """
        :type n: int
        :rtype: int
        """
        if n <= 2: return n
        prev_prev, prev = 1, 2
        cur = None
        for i in range(n - 2):
            cur = prev + prev_prev
            prev, prev_prev = cur, prev
        return cur


# ============================================================================

# 71. Simplify Path
# Difficulty: Medium
# link: https://leetcode.com/problems/simplify-path/
# Companies: Facebook,Microsoft
# Categories: String,Stack

# ----------------------------------------------------------------------------

class Solution(object):
    def simplifyPath(self, path):
        """
        :type path: str
        :rtype: str
        """
        queue = []
        path = filter(lambda x: x != '', path.split("/")[1:])
        while path:
            next_folder = path.pop(0)
            if next_folder == '.':
                pass
            elif next_folder == '..':
                if queue:
                    queue.pop()
            else:
                queue.append(next_folder)
        return '/' + '/'.join(queue)


# ============================================================================

# 73. Set Matrix Zeroes
# Difficulty: Medium
# link: https://leetcode.com/problems/set-matrix-zeroes/
# Companies: Amazon,Facebook,Microsoft
# Categories: Array

# ----------------------------------------------------------------------------

class Solution(object):
    def setZeroes(self, matrix):
        m, n = len(matrix), len(matrix[0])
        for i in range(m):
            for j in range(n):
                if matrix[i][j] == 0:
                    for x in range(m):
                        matrix[x][j] = None if matrix[x][j] else matrix[x][j]
                    for y in range(n):
                        matrix[i][y] = None if matrix[i][y] else matrix[i][y]
        for i in range(m):
            for j in range(n):
                matrix[i][j] = matrix[i][j] or 0


# ============================================================================

# 74. Search a 2D Matrix
# Difficulty: Medium
# link: https://leetcode.com/problems/search-a-2d-matrix/
# Companies: Amazon,Adobe,Microsoft
# Categories: Array,Binary Search

# ----------------------------------------------------------------------------

class Solution(object):
    def searchMatrix(self, matrix, target):
        """
        :type matrix: List[List[int]]
        :type target: int
        :rtype: bool
        """
        def _searchMatrix(start, end):
            if end <= 0:
                return False
            elif end == start + 1:
                return matrix[start/len(matrix[0])][start%len(matrix[0])] == target
            else:
                mid = (start + end) / 2
                mid_val = matrix[mid/len(matrix[0])][mid%len(matrix[0])]
                if mid_val == target:
                    return True
                elif mid_val < target:
                    return _searchMatrix(mid, end)
                else:
                    return _searchMatrix(start, mid)
        if not matrix:
            return False
        return _searchMatrix(0, len(matrix[0]) * len(matrix))


# ============================================================================

# 75. Sort Colors
# Difficulty: Medium
# link: https://leetcode.com/problems/sort-colors/
# Companies: Amazon,Google,Facebook,Microsoft,Apple
# Categories: Array,Two Pointers,Sort

# ----------------------------------------------------------------------------

class Solution(object):
    def sortColors(self, nums):
        def _sort(i, val):
            j = i
            while j < len(nums) and i < len(nums):
                if nums[i] == val: i += 1
                elif nums[max(i, j)] != val: j = max(i, j) + 1
                else:
                    nums[i], nums[j] = nums[j], nums[i]
            return i
        _sort(_sort(0, 0), 1)
        return nums


# ============================================================================

# 76. Minimum Window Substring
# Difficulty: Hard
# link: https://leetcode.com/problems/minimum-window-substring/
# Companies: Uber,GoDaddy,Google,Adobe,LinkedIn,Goldman Sachs,Amazon,Lyft,Facebook,Snapchat,Deutsche Bank,Microsoft
# Categories: Hash Table,Two Pointers,String,Sliding Window

# ----------------------------------------------------------------------------

class Solution(object):
    def minWindow(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: str
        """
    def minWindow(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: str
        """
        from collections import Counter
        start = 0
        t_counts = Counter(t)
        missing_count = len(t_counts)
        i, j = 0, float('inf')
        for end, c in enumerate(s):
            t_counts[c] -= 1
            if t_counts[c] == 0: missing_count -= 1
            while missing_count == 0:
                if end - start < j - i:
                    i, j = start, end + 1
                t_counts[s[start]] += 1
                if t_counts[s[start]] == 1: missing_count += 1
                start += 1
        return s[i: j] if j != float('inf') else ''


# ============================================================================

# 77. Combinations
# Difficulty: Medium
# link: https://leetcode.com/problems/combinations/
# Companies: Google,Microsoft,Apple
# Categories: Backtracking

# ----------------------------------------------------------------------------

class Solution(object):
    def combine(self, n, k):
        """
        :type n: int
        :type k: int
        :rtype: List[List[int]]
        """

        cur, res = [], []
        def _combinations(i):
            if len(cur) == k: return res.append(cur[:])
            for j in range(i, n + 1):
                cur.append(j)
                _combinations(j + 1)
                cur.pop()
        _combinations(1)
        return res


# ============================================================================

# 78. Subsets
# Difficulty: Medium
# link: https://leetcode.com/problems/subsets/
# Companies: Google,Adobe,Amazon,Facebook,Bloomberg,Walmart Labs,Microsoft
# Categories: Array,Backtracking,Bit Manipulation

# ----------------------------------------------------------------------------

class Solution(object):
    def subsets(self, nums): return [[nums[j] for j in range(len(nums)) if i & (1 << j)] for i in range(1 << len(nums))]


# ============================================================================

# 79. Word Search
# Difficulty: Medium
# link: https://leetcode.com/problems/word-search/
# Companies: Uber,Google,Apple,Intuit,Snapchat,Yahoo,Amazon,Facebook,Bloomberg,Oracle,Microsoft
# Categories: Array,Backtracking

# ----------------------------------------------------------------------------

class Solution(object):
    def exist(self, board, word):
        """
        :type board: List[List[str]]
        :type word: str
        :rtype: bool
        """
        m, n = len(board), len(board[0])
        visited = [[False] * n for _ in range(m)]
        def exist(i, j, char_i):
            if word[char_i] != board[i][j]: return False
            elif char_i == len(word) - 1 and not visited[i][j]: return True
            adjs = [(i + a, j + b)
                    for a, b in zip([1,0,-1,0], [0,1,0,-1])
                    if (0 <= i + a < m) and (0 <= j + b < n)]
            visited[i][j] = True
            if any(not visited[x][y] and exist(x, y, char_i + 1)
                   for x, y in adjs):
                    return True
            visited[i][j] = False
            return False

        return any(exist(i, j, 0)
                   for i in range(m)
                   for j in range(n))


# ============================================================================

# 80. Remove Duplicates from Sorted Array II
# Difficulty: Medium
# link: https://leetcode.com/problems/remove-duplicates-from-sorted-array-ii/
# Companies: Baidu,Facebook,Google,Adobe,Bloomberg
# Categories: Array,Two Pointers

# ----------------------------------------------------------------------------

class Solution(object):
    def removeDuplicates(self, nums):
        i = 0
        for num in nums:
            if i < 2 or num != nums[i - 2]:
                nums[i] = num
                i += 1
        return i


# ============================================================================

# 81. Search in Rotated Sorted Array II
# Difficulty: Medium
# link: https://leetcode.com/problems/search-in-rotated-sorted-array-ii/
# Companies: Facebook,Microsoft
# Categories: Array,Binary Search

# ----------------------------------------------------------------------------

class Solution(object):
    def search(self, nums, target):
        if not nums: return False
        # find lowest_idx
        lo, hi = 0, len(nums) - 1
        while lo <= hi:
            while (hi - lo) >= 1 and nums[hi] == nums[hi - 1]: hi -= 1
            # while (hi - lo) >= 1 and nums[lo] == nums[lo + 1]: lo += 1
            mid = (lo + hi) / 2
            if nums[mid] == target: return True
            elif nums[lo] <= target < nums[mid] or \
                (nums[lo] > nums[mid] and (target < nums[mid] or target >= nums[lo])): hi = mid - 1
            else: lo = mid + 1

        return False


# ============================================================================

# 82. Remove Duplicates from Sorted List II
# Difficulty: Medium
# link: https://leetcode.com/problems/remove-duplicates-from-sorted-list-ii/
# Companies: Amazon,Microsoft,Bloomberg
# Categories: Linked List

# ----------------------------------------------------------------------------

# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def deleteDuplicates(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        prev = dummy = ListNode('dummy')
        cur = dummy.next = head
        while cur and cur.next:
            if cur.val == cur.next.val:
                next_diff = cur
                while next_diff.next and next_diff.val == next_diff.next.val:
                    next_diff = next_diff.next
                prev.next = cur = next_diff.next
            else:
                prev, cur = cur, cur.next
        return dummy.next


# ============================================================================

# 83. Remove Duplicates from Sorted List
# Difficulty: Easy
# link: https://leetcode.com/problems/remove-duplicates-from-sorted-list/
# Companies: Amazon,Microsoft
# Categories: Linked List

# ----------------------------------------------------------------------------

# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def deleteDuplicates(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        cur = head
        res = res_head = ListNode('dummy')
        while cur:
            if cur.val != res.val:
                res.next = cur
                res = res.next
            cur = cur.next
        res.next = None
        return res_head.next


# ============================================================================

# 84. Largest Rectangle in Histogram
# Difficulty: Hard
# link: https://leetcode.com/problems/largest-rectangle-in-histogram/
# Companies: Microsoft,Bloomberg,Flipkart
# Categories: Array,Stack

# ----------------------------------------------------------------------------

class Solution(object):
    def largestRectangleArea(self, heights):
        """
        :type heights: List[int]
        :rtype: int
        """
        heights.append(0)
        dp = []
        max_area = 0
        for i, height in enumerate(heights):
            left = i
            while dp and dp[-1][1] > height:
                left = dp[-1][0]
                j, j_height = dp.pop()
                max_area = max(max_area, j_height * (i - j))
            dp.append((left, height))
        return max_area


# ============================================================================

# 87. Scramble String
# Difficulty: Hard
# link: https://leetcode.com/problems/scramble-string/
# Companies: Google
# Categories: String,Dynamic Programming

# ----------------------------------------------------------------------------

class Solution(object):
    def isScramble(self, s1, s2):
        """
        :type s1: str
        :type s2: str
        :rtype: bool
        """
        dp = {}
        from collections import Counter
        def _isScramble(s1, s2):
            if s1 == s2: return True
            elif (s1, s2) in dp: return dp[(s1, s2)]
            elif sorted(s1) != sorted(s2):
                dp[s1, s2] = False
                return False
            n = len(s1)
            f = _isScramble
            for i in range(1, len(s1)):
                if f(s1[i:], s2[i:]) and f(s1[:i], s2[:i]) or \
                   f(s1[i:], s2[:-i]) and f(s1[:i], s2[-i:]):
                    dp[(s1, s2)] = True
                    return True
            dp[(s1, s2)] = False
            return False
        return _isScramble(s1, s2)


# ============================================================================

# 88. Merge Sorted Array
# Difficulty: Easy
# link: https://leetcode.com/problems/merge-sorted-array/
# Companies: Google,Cisco,Apple,Yahoo,LinkedIn,eBay,Amazon,Facebook,Yandex,Bloomberg,Adobe,Microsoft
# Categories: Array,Two Pointers

# ----------------------------------------------------------------------------

class Solution(object):
    def merge(self, nums1, m, nums2, n):
        """
        :type nums1: List[int]
        :type m: int
        :type nums2: List[int]
        :type n: int
        :rtype: void Do not return anything, modify nums1 in-place instead.
        """
        i, j = m - 1, n - 1
        for k in range(i + j + 1, -1, -1):
            if j < 0: nums1[k], i = nums1[i], i - 1
            elif i < 0: nums1[k], j = nums2[j], j - 1
            elif nums1[i] > nums2[j]: nums1[k], i = nums1[i], i - 1
            else: nums1[k], j = nums2[j], j - 1


# ============================================================================

# 89. Gray Code
# Difficulty: Medium
# link: https://leetcode.com/problems/gray-code/
# Companies: Amazon,Google
# Categories: Backtracking

# ----------------------------------------------------------------------------

class Solution(object):
    def grayCode(self, n):
        """
        :type n: int
        :rtype: List[int]
        """
        gray = [0]
        for i in xrange(n):
            tog = (1 << i)
            gray.extend([g | tog for g in reversed(gray)])
        return gray


# ============================================================================

# 90. Subsets II
# Difficulty: Medium
# link: https://leetcode.com/problems/subsets-ii/
# Companies: Amazon,Microsoft
# Categories: Array,Backtracking

# ----------------------------------------------------------------------------

class Solution(object):
    def subsetsWithDup(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        from collections import Counter
        counts = Counter(nums)
        uniq = counts.keys()
        cur, res = [], [[]]
        def _subsetsWithDup(idx=0):
            if idx >= len(uniq): return
            if counts[uniq[idx]]:
                counts[uniq[idx]] -= 1
                cur.append(uniq[idx])
                res.append(cur[:])
                _subsetsWithDup(idx)
                cur.pop()
                counts[uniq[idx]] += 1
            _subsetsWithDup(idx + 1)
        _subsetsWithDup()
        return res


# ============================================================================

# 92. Reverse Linked List II
# Difficulty: Medium
# link: https://leetcode.com/problems/reverse-linked-list-ii/
# Companies: Amazon,Microsoft
# Categories: Linked List

# ----------------------------------------------------------------------------

# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution(object):
    def reverseBetween(self, head, m, n):
        """
        :type head: ListNode
        :type m: int
        :type n: int
        :rtype: ListNode
        """
        dummy = cur = ListNode(0)
        cur.next = head
        for i in range(m): prev, cur = cur, cur.next
        tail1, tail2 = prev, cur
        prev = None
        for i in range(n - m + 1):
            cur.next, prev, cur = prev, cur, cur.next
            # tmp = cur.next
            # cur.next = prev
            # prev, cur = cur, tmp
        tail1.next, tail2.next = prev, cur
        return dummy.next


# ============================================================================

# 93. Restore IP Addresses
# Difficulty: Medium
# link: https://leetcode.com/problems/restore-ip-addresses/
# Companies: Amazon,Microsoft,VMware
# Categories: String,Backtracking

# ----------------------------------------------------------------------------

class Solution(object):
    def restoreIpAddresses(self, s):
        """
        :type s: str
        :rtype: List[str]
        """
        n = len(s)
        if float(n) / 4 > 3: return []
        dp = [[] for _ in xrange(n + 1)]
        dp[0].append([])
        for j in range(1, n + 1):
            for i in range(max(j - 3, 0), j):
                for ip in dp[i]:
                    if ((s[i:j] == '0') or \
                        (not s[i:j].startswith('0') and int(s[i:j]) < 256)) \
                         and len(ip) < 4:
                        dp[j].append(ip + [s[i:j]])

        return ['.'.join(ip) for ip in dp[n] if len(ip) == 4]


# ============================================================================

# 94. Binary Tree Inorder Traversal
# Difficulty: Medium
# link: https://leetcode.com/problems/binary-tree-inorder-traversal/
# Companies: SAP,Amazon,Google,Facebook,Microsoft
# Categories: Hash Table,Stack,Tree

# ----------------------------------------------------------------------------

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def inorderTraversal(self, root):
        stack = []
        def move_left(node):
            while node:
                stack.append(node)
                node = node.left
        move_left(root)
        res = []
        while stack:
            next_elem = stack.pop()
            res.append(next_elem.val)
            if next_elem.right: move_left(next_elem.right)
        return res


# ============================================================================

# 95. Unique Binary Search Trees II
# Difficulty: Medium
# link: https://leetcode.com/problems/unique-binary-search-trees-ii/
# Companies: Microsoft
# Categories: Dynamic Programming,Tree

# ----------------------------------------------------------------------------

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def generateTrees(self, n):
        """
        :type n: int
        :rtype: List[TreeNode]
        """
        lst = range(1, n + 1)
        if not n: return []
        def clone(node):
            if not node: return
            new_node = TreeNode(node.val)
            new_node.left = clone(node.left)
            new_node.right = clone(node.right)
            return new_node
        def _generateTrees(i, j):
            if i >= j: return [None]
            res = []
            for k in range(i, j):
                left = _generateTrees(i, k)
                right = _generateTrees(k + 1, j)
                for l in left:
                    for r in right:
                        node = TreeNode(lst[k])
                        node.left = clone(l)
                        node.right = clone(r)
                        res.append(node)
            return res
        return _generateTrees(0, len(lst))


# ============================================================================

# 96. Unique Binary Search Trees
# Difficulty: Medium
# link: https://leetcode.com/problems/unique-binary-search-trees/
# Companies: Amazon
# Categories: Dynamic Programming,Tree

# ----------------------------------------------------------------------------

class Solution(object):
    def numTrees(self, n):
        """
        :type n: int
        :rtype: int
        """
        dp = [1, 1]
        if n < len(dp):
            return dp[n]
        for i in xrange(2, n + 1):
            next_val = 0
            for j in xrange(1, i + 1):
                next_val += (dp[j - 1]) * (dp[i - j])
            dp.append(next_val)
        return dp[-1]


# ============================================================================

# 97. Interleaving String
# Difficulty: Hard
# link: https://leetcode.com/problems/interleaving-string/
# Companies: Amazon,Google,Apple,Microsoft
# Categories: String,Dynamic Programming

# ----------------------------------------------------------------------------

class Solution(object):
    def isInterleave(self, s1, s2, s3):
        if len(s1) + len(s2) != len(s3): return False

        m, n = len(s1), len(s2)
        DP = [[False] * (n + 1) for _ in range(m + 1)]


        for i in range(m + 1):
            for j in range(n + 1):
                if i == j == 0: DP[i][j] = True
                elif j == 0:
                    DP[i][j] = DP[i - 1][0] and s1[i - 1] == s3[i - 1]
                elif i == 0:
                    DP[i][j] = DP[0][j - 1] and s2[j - 1] == s3[j - 1]
                else:
                    DP[i][j] = (DP[i-1][j] and s1[i - 1] == s3[i + j - 1]) \
                            or (DP[i][j-1] and s2[j - 1] == s3[i + j - 1])
        return DP[-1][-1]


# ============================================================================

# 98. Validate Binary Search Tree
# Difficulty: Medium
# link: https://leetcode.com/problems/validate-binary-search-tree/
# Companies: Google,Adobe,Apple,Salesforce,Amazon,Asana,Facebook,Yandex,Bloomberg,Microsoft
# Categories: Tree,Depth-first Search

# ----------------------------------------------------------------------------

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def isValidBST(self, root):
        def _isValidBST(node, min_val, max_val):
            if not node: return True
            if min_val < node.val < max_val and _isValidBST(node.left, min_val, node.val) and _isValidBST(node.right, node.val, max_val):
                return True
            return False
        return _isValidBST(root, float('-inf'), float('inf'))


# ============================================================================

# 100. Same Tree
# Difficulty: Easy
# link: https://leetcode.com/problems/same-tree/
# Companies: Amazon,Google,Microsoft
# Categories: Tree,Depth-first Search

# ----------------------------------------------------------------------------

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def isSameTree(self, p, q):
        if (p and not q) or (not p and q): return False
        elif not p and not q: return True
        return p.val == q.val and self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right)


# ============================================================================

# 101. Symmetric Tree
# Difficulty: Easy
# link: https://leetcode.com/problems/symmetric-tree/
# Companies: Uber,Google,Apple,Twitter,Atlassian,Amazon,Yandex,SAP,Microsoft
# Categories: Tree,Depth-first Search,Breadth-first Search

# ----------------------------------------------------------------------------

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def isSymmetric(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        def _isSymmetric(t1, t2):
            if not t1 and not t2: return True
            elif (not t1 and t2) or (t1 and not t2): return False
            return t1.val == t2.val and \
                    _isSymmetric(t1.left, t2.right) and \
                    _isSymmetric(t1.right, t2.left)
        if not root: return True
        return _isSymmetric(root.left, root.right)


# ============================================================================

# 102. Binary Tree Level Order Traversal
# Difficulty: Medium
# link: https://leetcode.com/problems/binary-tree-level-order-traversal/
# Companies: LinkedIn,Amazon,Facebook,Bloomberg,Walmart Labs,SAP,Microsoft,VMware
# Categories: Tree,Breadth-first Search

# ----------------------------------------------------------------------------

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def levelOrder(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        if not root: return []
        bfs = [root]
        res = []
        while bfs:
            res.append([node.val for node in bfs])
            bfs = [kid for node in bfs for kid in [node.left, node.right] if kid]
        return res


# ============================================================================

# 103. Binary Tree Zigzag Level Order Traversal
# Difficulty: Medium
# link: https://leetcode.com/problems/binary-tree-zigzag-level-order-traversal/
# Companies: Qualtrics,ByteDance,ServiceNow,Amazon,Facebook,Bloomberg,Microsoft
# Categories: Stack,Tree,Breadth-first Search

# ----------------------------------------------------------------------------

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def zigzagLevelOrder(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        if not root: return []
        bfs = [root]
        res = []
        while bfs:
            new_item = [node.val for node in bfs]
            res.append(new_item[::-1] if len(res) % 2 else new_item)
            bfs = [kid for node in bfs for kid in [node.left, node.right] if kid]
        return res


# ============================================================================

# 104. Maximum Depth of Binary Tree
# Difficulty: Easy
# link: https://leetcode.com/problems/maximum-depth-of-binary-tree/
# Companies: Amazon,Google,Facebook,Microsoft,LinkedIn
# Categories: Tree,Depth-first Search

# ----------------------------------------------------------------------------

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def maxDepth(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        if not root: return 0
        depth = 0
        bfs = [root]
        while bfs:
            depth += 1
            bfs = [kid for node in bfs for kid in [node.left, node.right] if kid]
        return depth


# ============================================================================

# 105. Construct Binary Tree from Preorder and Inorder Traversal
# Difficulty: Medium
# link: https://leetcode.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/
# Companies: Amazon,Square,Facebook,Microsoft,Apple
# Categories: Array,Tree,Depth-first Search

# ----------------------------------------------------------------------------

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def buildTree(self, preorder, inorder):
        """
        :type preorder: List[int]
        :type inorder: List[int]
        :rtype: TreeNode
        """
        def _buildTree(pre_s, pre_e, in_s, in_e):
            if pre_s >= pre_e or in_s >= in_e:
                return
            root = TreeNode(preorder[pre_s])
            idx = inorder.index(root.val, in_s, in_e)
            left_dist = idx - in_s
            root.left = _buildTree(pre_s + 1, pre_s + 1 + left_dist, in_s, idx)
            right_dist = in_e - idx - 1
            root.right = _buildTree(pre_s + 1 + left_dist, pre_s + 1 + left_dist + right_dist, idx + 1, idx + 1 + right_dist)
            return root

        return _buildTree(0, len(preorder), 0, len(preorder))


# ============================================================================

# 106. Construct Binary Tree from Inorder and Postorder Traversal
# Difficulty: Medium
# link: https://leetcode.com/problems/construct-binary-tree-from-inorder-and-postorder-traversal/
# Companies: Amazon
# Categories: Array,Tree,Depth-first Search

# ----------------------------------------------------------------------------

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def buildTree(self, inorder, postorder):
        """
        :type inorder: List[int]
        :type postorder: List[int]
        :rtype: TreeNode
        """
        def _buildTree(i_l, i_r):
            if i_l >= i_r: return
            node = TreeNode(postorder.pop())
            elem_idx = inorder.index(node.val)
            node.right = _buildTree(elem_idx + 1, i_r)
            node.left = _buildTree(i_l, elem_idx)
            return node
        return _buildTree(0, len(inorder))


# ============================================================================

# 107. Binary Tree Level Order Traversal II
# Difficulty: Easy
# link: https://leetcode.com/problems/binary-tree-level-order-traversal-ii/
# Companies: Adobe,Microsoft
# Categories: Tree,Breadth-first Search

# ----------------------------------------------------------------------------

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def levelOrderBottom(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        if not root: return []
        lvl_tra = []
        bfs = [root]
        while bfs:
            lvl_tra.append([node.val for node in bfs])
            bfs = [kid for node in bfs for kid in [node.left, node.right] if kid]
        return lvl_tra[::-1]


# ============================================================================

# 108. Convert Sorted Array to Binary Search Tree
# Difficulty: Easy
# link: https://leetcode.com/problems/convert-sorted-array-to-binary-search-tree/
# Companies: Amazon,Microsoft
# Categories: Tree,Depth-first Search

# ----------------------------------------------------------------------------

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def sortedArrayToBST(self, nums):
        """
        :type nums: List[int]
        :rtype: TreeNode
        """
        def _sortedArrayToBST(i, j):
            if i >= j: return
            mid = (i + j) / 2
            node = TreeNode(nums[mid])
            node.left = _sortedArrayToBST(i, mid)
            node.right = _sortedArrayToBST(mid + 1, j)
            return node

        return _sortedArrayToBST(0, len(nums))


# ============================================================================

# 109. Convert Sorted List to Binary Search Tree
# Difficulty: Medium
# link: https://leetcode.com/problems/convert-sorted-list-to-binary-search-tree/
# Companies: Oracle,Amazon,Google,Facebook
# Categories: Linked List,Depth-first Search

# ----------------------------------------------------------------------------

# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def sortedListToBST(self, head):
        """
        :type head: ListNode
        :rtype: TreeNode
        """
        lst = []
        cur = head
        while cur:
            lst.append(cur.val)
            cur = cur.next

        def _sortedListToBST(start, end):
            if start >= end:
                return
            mid = (end + start) / 2
            node = TreeNode(lst[mid])
            node.left = _sortedListToBST(start, mid)
            node.right = _sortedListToBST(mid + 1, end)
            return node

        return _sortedListToBST(0, len(lst))


# ============================================================================

# 110. Balanced Binary Tree
# Difficulty: Easy
# link: https://leetcode.com/problems/balanced-binary-tree/
# Companies: Google,Facebook,Microsoft
# Categories: Tree,Depth-first Search

# ----------------------------------------------------------------------------

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def isBalanced(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        def _isBalanced(node):
            if not node: return 1
            left = _isBalanced(node.left)
            if left == -1: return -1
            right = _isBalanced(node.right)
            return -1 if (abs(left - right) > 1 or right == -1) else max(left, right) + 1
        return _isBalanced(root) != -1


# ============================================================================

# 111. Minimum Depth of Binary Tree
# Difficulty: Easy
# link: https://leetcode.com/problems/minimum-depth-of-binary-tree/
# Companies: Facebook
# Categories: Tree,Depth-first Search,Breadth-first Search

# ----------------------------------------------------------------------------

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def minDepth(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        if not root: return 0
        depth, bfs = 0, [root]
        while bfs:
            depth += 1
            if next((True for node in bfs if not node.left and not node.right), False): return depth
            bfs = [kid for node in bfs for kid in [node.left, node.right] if kid]


# ============================================================================

# 112. Path Sum
# Difficulty: Easy
# link: https://leetcode.com/problems/path-sum/
# Companies: Amazon,Facebook,Microsoft
# Categories: Tree,Depth-first Search

# ----------------------------------------------------------------------------

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def hasPathSum(self, root, total):
        cur_path = []
        def _pathSum(node, sum_from_root):
            if not node:
                return False
            sum_from_root += node.val
            cur_path.append(node.val)
            if sum_from_root == total and not node.left and not node.right:
                return True
            res = _pathSum(node.left, sum_from_root) or _pathSum(node.right, sum_from_root)
            cur_path.pop()
            return res
        return _pathSum(root, 0)


# ============================================================================

# 113. Path Sum II
# Difficulty: Medium
# link: https://leetcode.com/problems/path-sum-ii/
# Companies: Amazon,Google,Facebook,Zillow
# Categories: Tree,Depth-first Search

# ----------------------------------------------------------------------------

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def pathSum(self, root, total):
        """
        :type root: TreeNode
        :type sum: int
        :rtype: List[List[int]]
        """
        cur_path = []
        res = []

        def _pathSum(node, sum_from_root):
            if not node:
                return
            sum_from_root += node.val
            cur_path.append(node.val)
            if sum_from_root == total and not node.left and not node.right:
                res.append(cur_path[:])
            _pathSum(node.left, sum_from_root)
            _pathSum(node.right, sum_from_root)
            cur_path.pop()
        _pathSum(root, 0)
        return res


# ============================================================================

# 114. Flatten Binary Tree to Linked List
# Difficulty: Medium
# link: https://leetcode.com/problems/flatten-binary-tree-to-linked-list/
# Companies: Facebook,Microsoft,Bloomberg
# Categories: Tree,Depth-first Search

# ----------------------------------------------------------------------------

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def flatten(self, root):
        """
        :type root: TreeNode
        :rtype: void Do not return anything, modify root in-place instead.
        """
        def _flatten(node):
            if not node:
                return
            flatten_left = _flatten(node.left)
            l_end = None
            if flatten_left:
                l_start, l_end = flatten_left
                node.right, l_end.right, node.left = l_start, node.right, None
                flatten_right = _flatten(l_end.right)
            else:
                flatten_right = _flatten(node.right)
            r_start, r_end = flatten_right if flatten_right else [None, None]

            if r_end:
                return node, r_end
            elif l_end:
                return node, l_end
            else:
                return node, node
        _flatten(root)


# ============================================================================

# 116. Populating Next Right Pointers in Each Node
# Difficulty: Medium
# link: https://leetcode.com/problems/populating-next-right-pointers-in-each-node/
# Companies: Oracle,Amazon,Google,Microsoft,Bloomberg
# Categories: Tree,Depth-first Search

# ----------------------------------------------------------------------------

# Definition for binary tree with next pointer.
# class TreeLinkNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
#         self.next = None

class Solution:
    # @param root, a tree link node
    # @return nothing
    def connect(self, root):
        if not root:
            return
        lvl = root
        while lvl.left:
            cur = lvl
            while cur:
                cur.left.next = cur.right
                if cur.next:
                    cur.right.next = cur.next.left
                cur = cur.next
            lvl = lvl.left


# ============================================================================

# 117. Populating Next Right Pointers in Each Node II
# Difficulty: Medium
# link: https://leetcode.com/problems/populating-next-right-pointers-in-each-node-ii/
# Companies: Oracle,Amazon,Google,Microsoft,Bloomberg
# Categories: Tree,Depth-first Search

# ----------------------------------------------------------------------------

# Definition for binary tree with next pointer.
# class TreeLinkNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
#         self.next = None

class Solution:
    # @param root, a tree link node
    # @return nothing
    def connect(self, root):
        lvl_head = root
        while lvl_head:
            cur = lvl_head
            next_lvl_head = next_lvl_cur = TreeLinkNode(-1)
            while cur:
                if cur.left:
                    next_lvl_cur.next = cur.left
                    next_lvl_cur = next_lvl_cur.next
                if cur.right:
                    next_lvl_cur.next = cur.right
                    next_lvl_cur = next_lvl_cur.next
                cur = cur.next
            lvl_head = next_lvl_head.next


# ============================================================================

# 118. Pascal's Triangle
# Difficulty: Easy
# link: https://leetcode.com/problems/pascals-triangle/
# Companies: Amazon,Microsoft,Goldman Sachs
# Categories: Array

# ----------------------------------------------------------------------------

class Solution(object):
    def generate(self, numRows):
        """
        :type numRows: int
        :rtype: List[List[int]]
        """
        tri = []
        for i in range(1, numRows + 1):
            for j in range(i):
                if j == 0: tri.append([1])
                elif j == i - 1: tri[-1].append(1)
                else: tri[-1].append(sum(tri[-2][j - 1: j + 1]))
        return tri


# ============================================================================

# 119. Pascal's Triangle II
# Difficulty: Easy
# link: https://leetcode.com/problems/pascals-triangle-ii/
# Companies: Qualtrics,Goldman Sachs
# Categories: Array

# ----------------------------------------------------------------------------

class Solution(object):
    def getRow(self, rowIndex):
        """
        :type rowIndex: int
        :rtype: List[int]
        """
        if rowIndex <= 1: return [1] * (rowIndex + 1)
        cur_row = [1, 1]
        for i in range(rowIndex - 1):
            cur_row = [1] + [sum([cur_row[i-1], cur_row[i]]) for i in range(1, len(cur_row))] + [1]
        return cur_row


# ============================================================================

# 120. Triangle
# Difficulty: Medium
# link: https://leetcode.com/problems/triangle/
# Companies: Amazon,Apple,Bloomberg
# Categories: Array,Dynamic Programming

# ----------------------------------------------------------------------------

class Solution(object):
    def minimumTotal(self, triangle):
        """
        :type triangle: List[List[int]]
        :rtype: int
        """
        for i in xrange(len(triangle) - 2, -1, -1):
            row = triangle[i]
            next_row = triangle[i + 1]
            for j in xrange(len(row)):
                row[j] += min(next_row[j], next_row[j + 1])
        return triangle[0][0] if triangle else 0


# ============================================================================

# 121. Best Time to Buy and Sell Stock
# Difficulty: Easy
# link: https://leetcode.com/problems/best-time-to-buy-and-sell-stock/
# Companies: Uber,Morgan Stanley,Google,Adobe,Apple,JPMorgan,Goldman Sachs,Amazon,Visa,Facebook,Bloomberg,Oracle,Tencent,SAP,DoorDash,Microsoft,Intel
# Categories: Array,Dynamic Programming

# ----------------------------------------------------------------------------

class Solution(object):
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        max_profit = max_so_far = 0
        for price in reversed(prices):
            max_profit = max(max_so_far - price, max_profit)
            max_so_far = max(max_so_far, price)
        return max_profit


# ============================================================================

# 122. Best Time to Buy and Sell Stock II
# Difficulty: Easy
# link: https://leetcode.com/problems/best-time-to-buy-and-sell-stock-ii/
# Companies: Uber,Amazon,Google,Facebook,Apple
# Categories: Array,Greedy

# ----------------------------------------------------------------------------

class Solution(object):
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        profit = 0
        for i in xrange(1, len(prices)):
            profit += prices[i] - prices[i - 1] if prices[i] > prices[i - 1] else 0
        return profit


# ============================================================================

# 124. Binary Tree Maximum Path Sum
# Difficulty: Hard
# link: https://leetcode.com/problems/binary-tree-maximum-path-sum/
# Companies: Google,Apple,Amazon,Facebook,Bloomberg,Microsoft
# Categories: Tree,Depth-first Search

# ----------------------------------------------------------------------------

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def maxPathSum(self, root):

        self.max_sum = float('-inf')

        def _getmax(node):
            if not node: return 0
            l, r = _getmax(node.left), _getmax(node.right)
            self.max_sum = max(self.max_sum, l + r + node.val)
            return max(max(l, r) + node.val, 0)

        _getmax(root)
        return self.max_sum


# ============================================================================

# 125. Valid Palindrome
# Difficulty: Easy
# link: https://leetcode.com/problems/valid-palindrome/
# Companies: Google,Apple,Wish,LinkedIn,eBay,Wayfair,Amazon,Facebook,Yandex,Bloomberg,Microsoft
# Categories: Two Pointers,String

# ----------------------------------------------------------------------------

class Solution(object):
    def isPalindrome(self, s):
        """
        :type s: str
        :rtype: bool
        """
        s = [char.lower() for char in s if char.isalpha() or char.isdigit()]
        for i in xrange(len(s) / 2):
            if s[i] != s[-i-1]: return False
        return True


# ============================================================================

# 126. Word Ladder II
# Difficulty: Hard
# link: https://leetcode.com/problems/word-ladder-ii/
# Companies: Uber,Amazon,Google,Facebook
# Categories: Array,String,Backtracking,Breadth-first Search

# ----------------------------------------------------------------------------

class Solution(object):
    def findLadders(self, beginWord, endWord, wordList):
        """
        :type beginWord: str
        :type endWord: str
        :type wordList: List[str]
        :rtype: List[List[str]]
        """
        from collections import defaultdict
        n = len(beginWord)
        adj_dict = defaultdict(list)
        for word in wordList:
            for i in range(n):
                adj_dict[word[:i] + '?' + word[i + 1:]].append(word)
        bfs = [([beginWord], beginWord)]
        visited = {beginWord}
        while bfs:
            lvl_visited = set()
            bfs = [(path + [adj], adj)
                   for path, cur in bfs
                   for i in range(n)
                   for adj in adj_dict[cur[:i] + '?' + cur[i + 1:]]
                   if adj not in visited and (lvl_visited.add(adj) is None)]
            visited.update(lvl_visited)
            final_path = [path for path, word in bfs if word == endWord]
            if final_path: return final_path
        return []


# ============================================================================

# 127. Word Ladder
# Difficulty: Medium
# link: https://leetcode.com/problems/word-ladder/
# Companies: Uber,Lyft,Apple,LinkedIn,Amazon,Qualtrics,Facebook,Bloomberg,Zillow,Microsoft,Google
# Categories: Breadth-first Search

# ----------------------------------------------------------------------------

class Solution(object):
    def ladderLength(self, beginWord, endWord, wordList):
        """
        :type beginWord: str
        :type endWord: str
        :type wordList: List[str]
        :rtype: int
        """
        from collections import defaultdict
        n = len(beginWord)
        adj_dict = defaultdict(list)
        for word in wordList:
            for i in range(n):
                adj_dict[word[:i] + '?' + word[i + 1:]].append(word)
        bfs = [(1, beginWord)]
        visited = {beginWord}
        while bfs:
            bfs = [
                (count + 1, adj)
                for count, cur in bfs
                for i in range(n)
                for adj in adj_dict[cur[:i] + '?' + cur[i + 1:]]
                if adj not in visited and (visited.add(adj) is None)
            ]
            final_path = next((count for count, word in bfs if word == endWord), None)
            if final_path: return final_path
        return 0


# ============================================================================

# 128. Longest Consecutive Sequence
# Difficulty: Hard
# link: https://leetcode.com/problems/longest-consecutive-sequence/
# Companies: Uber,Wish,Amazon,Google,Microsoft
# Categories: Array,Union Find

# ----------------------------------------------------------------------------

class Solution(object):
    def longestConsecutive(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        nums = set(nums)
        res = 0
        for num in nums:
            if num - 1 not in nums:
                for i in range(len(nums)):
                    if num + i in nums: res = max(i + 1, res)
                    else: break

        return res


# ============================================================================

# 129. Sum Root to Leaf Numbers
# Difficulty: Medium
# link: https://leetcode.com/problems/sum-root-to-leaf-numbers/
# Companies: Amazon,Facebook
# Categories: Tree,Depth-first Search

# ----------------------------------------------------------------------------

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def sumNumbers(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        def _sumNumbers(node, cur_num=[]):
            if not node:
                return 0
            cur_num.append(node.val)
            if not node.left and not node.right:
                res = int(''.join(map(lambda x: str(x), cur_num)))
            else:
                res =  _sumNumbers(node.left, cur_num) + _sumNumbers(node.right, cur_num)
            cur_num.pop()
            return res
        return _sumNumbers(root)


# ============================================================================

# 130. Surrounded Regions
# Difficulty: Medium
# link: https://leetcode.com/problems/surrounded-regions/
# Companies: Amazon,Google,Splunk
# Categories: Depth-first Search,Breadth-first Search,Union Find

# ----------------------------------------------------------------------------

class Solution(object):
    def solve(self, board):
        """
        :type board: List[List[str]]
        :rtype: void Do not return anything, modify board in-place instead.
        """
        if not board: return
        m, n = len(board), len(board[0])
        bfs = [pair for x in range(m) for pair in [(x, 0),(x, n - 1)]] + \
              [pair for y in range(n) for pair in [(0, y),(m - 1, y)]]

        while bfs:
            x, y = bfs.pop()
            if 0 <= x < m and 0 <= y < n and board[x][y] == 'O':
                board[x][y] = ''
                bfs.extend([x + a, y + b] for a, b in zip([0,1,0,-1], [1,0,-1,0]))
        for i in range(m):
            for j in range(n):
                board[i][j] = 'X' if board[i][j] else 'O'


# ============================================================================

# 131. Palindrome Partitioning
# Difficulty: Medium
# link: https://leetcode.com/problems/palindrome-partitioning/
# Companies: Uber
# Categories: Backtracking

# ----------------------------------------------------------------------------

class Solution(object):
    def partition(self, s):
        """
        :type s: str
        :rtype: List[List[str]]
        """
        n = len(s) + 1
        # is_pa = [[False] * n for _ in xrange(n)]
        # for i in xrange(n):
        #     for j in xrange(i, n):
        #         if s[i:j] == s[i:j][::-1]:
        #             is_pa[i][j] = True
        dp = [[[]]]
        for j in xrange(1, n):
            dp.append([prefix + [s[i:j]]
                            for i in xrange(j)
                                if s[i:j] == s[i:j][::-1]
                                    for prefix in dp[i]])
        return dp[-1]


# ============================================================================

# 132. Palindrome Partitioning II
# Difficulty: Hard
# link: https://leetcode.com/problems/palindrome-partitioning-ii/
# Companies: Google
# Categories: Dynamic Programming

# ----------------------------------------------------------------------------

class Solution(object):
    def minCut(self, s):
        """
        :type s: str
        :rtype: int
        """
        n = len(s)
        is_pal = [[None] * (n + 1) for _ in range(n + 1)]
        for i in range(n): is_pal[i][i] = is_pal[i][i + 1] = True
        for k in range(2, n + 1):
            for i in range(n - k + 1):
                j = i + k
                is_pal[i][j] = s[i] == s[j - 1] and is_pal[i + 1][j - 1]
        dp = [float('inf')] * (n + 1)
        dp[0] = -1
        for j in range(n + 1):
            if j: dp[j] = dp[j - 1] + 1
            for i in range(j):
                if is_pal[i][j]: dp[j] = min(dp[j], dp[i] + 1)
        return dp[-1]


# ============================================================================

# 133. Clone Graph
# Difficulty: Medium
# link: https://leetcode.com/problems/clone-graph/
# Companies: Amazon,Google,Facebook,Microsoft,Apple
# Categories: Depth-first Search,Breadth-first Search,Graph

# ----------------------------------------------------------------------------

# Definition for a undirected graph node
# class UndirectedGraphNode:
#     def __init__(self, x):
#         self.label = x
#         self.neighbors = []

class Solution:
    # @param node, a undirected graph node
    # @return a undirected graph node
    def cloneGraph(self, node):
        start_node = node
        node_to_clone = {}
        def clone_if_not_exists(node):
            if node not in node_to_clone:
                node_to_clone[node] = UndirectedGraphNode(node.label)
        if not node:
            return None
        queue = [node]
        copied = set()
        while queue:
            node = queue.pop()
            if node not in copied:
                clone_if_not_exists(node)
                for neighbor in node.neighbors:
                    clone_if_not_exists(neighbor)
                    node_to_clone[node].neighbors.append(node_to_clone[neighbor])
                    queue.insert(0, neighbor)
                copied.add(node)
        return node_to_clone[start_node]


# ============================================================================

# 134. Gas Station
# Difficulty: Medium
# link: https://leetcode.com/problems/gas-station/
# Companies: Amazon,Expedia,Microsoft
# Categories: Greedy

# ----------------------------------------------------------------------------

class Solution(object):
    def canCompleteCircuit(self, gas, cost):
        """
        :type gas: List[int]
        :type cost: List[int]
        :rtype: int
        """
        diff = [gas[i] - cost[i] for i in range(len(gas))]
        if len(diff) == 1:
            if diff[0] >= 0:
                return 0
            else:
                return -1
        start_idx = 0
        end_idx = 0
        if not diff:
            return True
        # keep acc positive
        acc = diff[start_idx]
        for i in range(len(diff) - 1):
            if acc <= 0:
                start_idx -= 1
                acc += diff[start_idx % len(diff)]
            else:
                end_idx += 1
                acc += diff[end_idx % len(diff)]
        if acc >= 0:
            return start_idx % len(diff)
        else:
            return -1


# ============================================================================

# 136. Single Number
# Difficulty: Easy
# link: https://leetcode.com/problems/single-number/
# Companies: Amazon,Facebook,Apple,Bloomberg
# Categories: Hash Table,Bit Manipulation

# ----------------------------------------------------------------------------

class Solution(object):
    def singleNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        return reduce(lambda x, y: x ^ y, nums, 0)


# ============================================================================

# 139. Word Break
# Difficulty: Medium
# link: https://leetcode.com/problems/word-break/
# Companies: Google,TripAdvisor,Apple,Amazon,Qualtrics,Facebook,Bloomberg,Square,Microsoft
# Categories: Dynamic Programming

# ----------------------------------------------------------------------------

class Solution(object):
    def wordBreak(self, s, wordDict):
        """
        :type s: str
        :type wordDict: List[str]
        :rtype: bool
        """
        dp = [False] * (len(s) + 1)
        dp[0] = True
        wordDict = set(wordDict)
        for j in range(1, len(s) + 1):
            for i in range(j - 1, -1, -1):
                if dp[i] and s[i: j] in wordDict:
                    dp[j] = True
                    continue
        return dp[-1]


# ============================================================================

# 141. Linked List Cycle
# Difficulty: Easy
# link: https://leetcode.com/problems/linked-list-cycle/
# Companies: Google,Apple,Amazon,Walmart Labs,Microsoft,VMware
# Categories: Linked List,Two Pointers

# ----------------------------------------------------------------------------

# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def hasCycle(self, head):
        visited = set()
        cur = head
        while cur:
            if cur in visited: return True
            visited.add(cur)
            cur = cur.next
        return False


# ============================================================================

# 142. Linked List Cycle II
# Difficulty: Medium
# link: https://leetcode.com/problems/linked-list-cycle-ii/
# Companies: Amazon
# Categories: Linked List,Two Pointers

# ----------------------------------------------------------------------------

# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def detectCycle(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        if not head or not head.next: return
        slow, fast = head.next, head.next.next
        while (fast and fast.next) and slow != fast:
            slow = slow.next
            fast = fast.next.next
        if not fast or not fast.next: return
        cur = head
        in_loop_cur = fast
        while cur != in_loop_cur:
            cur = cur.next
            in_loop_cur = in_loop_cur.next
        return cur


# ============================================================================

# 144. Binary Tree Preorder Traversal
# Difficulty: Medium
# link: https://leetcode.com/problems/binary-tree-preorder-traversal/
# Companies: Amazon
# Categories: Stack,Tree

# ----------------------------------------------------------------------------

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def preorderTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        if not root: return []
        bfs = [root]
        def check_node(node):
            if type(node) == TreeNode: return [node.val, node.left, node.right]
            else: return [node]
        while any(type(node) == TreeNode for node in bfs):
            bfs = [kid for node in bfs for kid in check_node(node) if kid is not None]
        return bfs


# ============================================================================

# 145. Binary Tree Postorder Traversal
# Difficulty: Hard
# link: https://leetcode.com/problems/binary-tree-postorder-traversal/
# Companies: Facebook
# Categories: Stack,Tree

# ----------------------------------------------------------------------------

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def postorderTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        if not root: return []
        bfs = [root]
        while any(None if type(node) != TreeNode else node for node in bfs):
            bfs = [kid for node in bfs for kid in ([node.left, node.right, node.val] if type(node) == TreeNode else [node]) if kid is not None]
        return bfs


# ============================================================================

# 147. Insertion Sort List
# Difficulty: Medium
# link: https://leetcode.com/problems/insertion-sort-list/
# Companies: Microsoft,Bloomberg
# Categories: Linked List,Sort

# ----------------------------------------------------------------------------

# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def insertionSortList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        dummy = ListNode(-1000000)
        dummy.next = head
        boundary_prev, boundary = dummy, head
        while boundary:
            node = boundary
            boundary = boundary.next
            boundary_prev.next = boundary
            prev, cur, node.next = dummy, dummy.next, None
            while cur and cur != boundary and cur.val < node.val: prev, cur = cur, cur.next
            tmp = prev.next
            prev.next = node
            node.next = tmp
            if boundary_prev.next != boundary: boundary_prev = boundary_prev.next
        return dummy.next


# ============================================================================

# 148. Sort List
# Difficulty: Medium
# link: https://leetcode.com/problems/sort-list/
# Companies: Amazon,Google,Adobe,Microsoft
# Categories: Linked List,Sort

# ----------------------------------------------------------------------------

# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def sortList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        def _merge_sort(head):
            def _get_from_queue(h1, h2):
                if not h1 and not h2: return None, None, None
                elif not h1 or (h1 and h2 and h2.val < h1.val): return h2, h1, h2.next
                elif not h2 or (h1 and h2 and h2.val >= h1.val): return h1, h1.next, h2
            if not head or not head.next: return head
            slow = fast = head
            while fast and fast.next and fast.next.next: slow, fast = slow.next, fast.next.next
            h1, h2, slow.next = head, slow.next, None
            h1 = _merge_sort(h1)
            h2 = _merge_sort(h2)
            cur = dummy = ListNode('dummy')
            while cur:
                cur.next, h1, h2 = _get_from_queue(h1, h2)
                cur = cur.next
            return dummy.next
        return _merge_sort(head)


# ============================================================================

# 149. Max Points on a Line
# Difficulty: Hard
# link: https://leetcode.com/problems/max-points-on-a-line/
# Companies: Uber,Amazon,Google,Apple,LinkedIn
# Categories: Hash Table,Math

# ----------------------------------------------------------------------------

# Definition for a point.
class Point(object):
    def __init__(self, a=0, b=0):
        self.x = a
        self.y = b
class Solution(object):
    def maxPoints(self, points):
        import numpy as np
        max_count = 0
        for i, point1 in enumerate(points):
            x1, y1 = point1.x, point1.y
            slope_cnt = {}
            same = 0
            for j, point2 in enumerate(points):
                if i != j:
                    x2, y2 = point2.x, point2.y
                    if (x1, y1) != (x2, y2):
                        slope = np.longdouble(y2 - y1) / (x2 - x1) if (x2 != x1) else 'inf'
                        slope_cnt[slope] = slope_cnt.get(slope, 1) + 1
                    else:
                        same += 1
            max_count = max(max(slope_cnt.values() or [1]) + same, max_count)
        return max_count or int(bool(points))


# ============================================================================

# 151. Reverse Words in a String
# Difficulty: Medium
# link: https://leetcode.com/problems/reverse-words-in-a-string/
# Companies: Amazon,Facebook,Microsoft
# Categories: String

# ----------------------------------------------------------------------------

class Solution(object):
    def reverseWords(self, s):
        """
        :type s: str
        :rtype: str
        """
        return " ".join(list(reversed(filter(lambda x: x != "", s.split(" ")))))


# ============================================================================

# 152. Maximum Product Subarray
# Difficulty: Medium
# link: https://leetcode.com/problems/maximum-product-subarray/
# Companies: Uber,Amazon,Google,LinkedIn
# Categories: Array,Dynamic Programming

# ----------------------------------------------------------------------------

class Solution(object):
    def maxProduct(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        large = small = max_val = nums[0]
        for i in range(1, len(nums)):
            num = nums[i]
            vals = [num, small * num, large * num]
            small, large = min(vals), max(vals)
            max_val = max(large, max_val)
        return max_val


# ============================================================================

# 153. Find Minimum in Rotated Sorted Array
# Difficulty: Medium
# link: https://leetcode.com/problems/find-minimum-in-rotated-sorted-array/
# Companies: Salesforce,Goldman Sachs,Amazon,Walmart Labs,Facebook,Microsoft
# Categories: Array,Binary Search

# ----------------------------------------------------------------------------

class Solution(object):
    def findMin(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        low, high = 0, len(nums)
        while low < high:
            mid = (low + high) / 2
            if mid + 1 >= len(nums):
                return min(nums[-1], nums[0])
            elif nums[mid] > nums[mid + 1]:
                return nums[mid + 1]
            elif nums[mid] < nums[0]:
                high = mid
            elif nums[mid] > nums[0]:
                low = mid


# ============================================================================

# 155. Min Stack
# Difficulty: Easy
# link: https://leetcode.com/problems/min-stack/
# Companies: Google,Adobe,Apple,Wish,Goldman Sachs,Amazon,Bloomberg,Microsoft,Flipkart
# Categories: Stack,Design

# ----------------------------------------------------------------------------

class MinStack(object):

    def __init__(self):
        self.lst = []
    def push(self, x):
        """
        :type x: int
        :rtype: void
        """
        cur_min = self.lst[-1][0] if self.lst else float('inf')
        self.lst.append([min(x, cur_min), x])

    def pop(self):
        """
        :rtype: void
        """
        self.lst.pop()

    def top(self):
        """
        :rtype: int
        """
        return self.lst[-1][1]

    def getMin(self):
        """
        :rtype: int
        """
        return self.lst[-1][0]


# Your MinStack object will be instantiated and called as such:
# obj = MinStack()
# obj.push(x)
# obj.pop()
# param_3 = obj.top()
# param_4 = obj.getMin()


# ============================================================================

# 160. Intersection of Two Linked Lists
# Difficulty: Easy
# link: https://leetcode.com/problems/intersection-of-two-linked-lists/
# Companies: Oracle,Yahoo,Microsoft,ByteDance,Bloomberg
# Categories: Linked List

# ----------------------------------------------------------------------------

# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def getIntersectionNode(self, headA, headB):
        """
        :type head1, head1: ListNode
        :rtype: ListNode
        """
        cur_a, cur_b = headA, headB
        while cur_a and cur_b: cur_a, cur_b = cur_a.next, cur_b.next
        longer, shorter = (headA, headB) if cur_a else (headB, headA)
        cur = cur_a or cur_b
        while cur: longer, cur = longer.next, cur.next
        while longer != shorter: longer, shorter = longer.next, shorter.next
        return shorter


# ============================================================================

# 162. Find Peak Element
# Difficulty: Medium
# link: https://leetcode.com/problems/find-peak-element/
# Companies: Uber,Google,Quora,Amazon,Facebook,Bloomberg,Microsoft
# Categories: Array,Binary Search

# ----------------------------------------------------------------------------

class Solution(object):
    def findPeakElement(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if not nums: return 0
        def get_val(idx):
            return nums[idx] if 0 <= idx < len(nums) else float('-inf')
        lo, hi = 0, len(nums) - 1
        while True:
            mid = (lo + hi) / 2
            if get_val(mid - 1) < get_val(mid) > get_val(mid + 1): return mid
            elif get_val(mid - 1) > get_val(mid): hi = mid - 1
            else: lo = mid + 1


# ============================================================================

# 163. Missing Ranges
# Difficulty: Medium
# link: https://leetcode.com/problems/missing-ranges/
# Companies: Google
# Categories: Array

# ----------------------------------------------------------------------------

class Solution(object):
    def findMissingRanges(self, nums, lower, upper):
        prev = None
        res = []
        from itertools import chain
        for num in chain([int(lower) - 1], nums, [int(upper) + 1]):
            if prev is not None and num - prev > 1:
                res.append(str(num - 1)
                           if prev == num - 2
                           else str(prev + 1) + '->' + str(num - 1))
            prev = num
        return res


# ============================================================================

# 164. Maximum Gap
# Difficulty: Hard
# link: https://leetcode.com/problems/maximum-gap/
# Companies: Amazon
# Categories: Sort

# ----------------------------------------------------------------------------

class Solution(object):
    def maximumGap(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        nums.sort()
        max_gap = 0
        for i in range(1, len(nums)):
            max_gap = max(nums[i] - nums[i - 1], max_gap)
        return max_gap


# ============================================================================

# 165. Compare Version Numbers
# Difficulty: Medium
# link: https://leetcode.com/problems/compare-version-numbers/
# Companies: Amazon
# Categories: String

# ----------------------------------------------------------------------------

class Solution(object):
    def compareVersion(self, version1, version2):
        """
        :type version1: str
        :type version2: str
        :rtype: int
        """
        version1 = [int(num) for num in version1.split('.')]
        version2 = [int(num) for num in version2.split('.')]
        while version1 and version1[-1] == 0:
            version1.pop()
        while version2 and version2[-1] == 0:
            version2.pop()

        for i in range(min(len(version1), len(version2))):
            if version1[i] < version2[i]:
                return -1
            elif version1[i] > version2[i]:
                return 1
        if len(version1) == len(version2):
            return 0
        elif len(version1) < len(version2):
            return -1
        else:
            return 1


# ============================================================================

# 167. Two Sum II - Input array is sorted
# Difficulty: Easy
# link: https://leetcode.com/problems/two-sum-ii-input-array-is-sorted/
# Companies: Amazon
# Categories: Array,Two Pointers,Binary Search

# ----------------------------------------------------------------------------

class Solution(object):
    def twoSum(self, numbers, target):
        """
        :type numbers: List[int]
        :type target: int
        :rtype: List[int]
        """
        i, j = 0, len(numbers) - 1
        while i < j:
            pair_sum = numbers[i] + numbers[j]
            if pair_sum == target: return i + 1, j + 1
            elif pair_sum < target: i += 1
            else: j -= 1


# ============================================================================

# 168. Excel Sheet Column Title
# Difficulty: Easy
# link: https://leetcode.com/problems/excel-sheet-column-title/
# Companies: Google,Microsoft
# Categories: Math

# ----------------------------------------------------------------------------

class Solution(object):
    def convertToTitle(self, n):
        """
        :type n: int
        :rtype: str
        """
        res = []
        mapping = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        while n != 0:
            res.append(mapping[(n - 1) % 26])
            n = (n - 1) / 26

        return ''.join(res[::-1])


# ============================================================================

# 169. Majority Element
# Difficulty: Easy
# link: https://leetcode.com/problems/majority-element/
# Companies: Amazon,Apple
# Categories: Array,Divide and Conquer,Bit Manipulation

# ----------------------------------------------------------------------------

class Solution(object):
    def majorityElement(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        count, most_freq_elem = 0, None
        for num in nums:
            if most_freq_elem is None:
                count, most_freq_elem = 1, num
            elif num != most_freq_elem:
                count -= 1
                if count == 0:
                    count, most_freq_elem = 1, num
            elif num == most_freq_elem:
                count += 1
        return most_freq_elem


# ============================================================================

# 171. Excel Sheet Column Number
# Difficulty: Easy
# link: https://leetcode.com/problems/excel-sheet-column-number/
# Companies: Amazon,Google,Microsoft
# Categories: Math

# ----------------------------------------------------------------------------

class Solution(object):
    def titleToNumber(self, s):
        """
        :type s: str
        :rtype: int
        """
        return reduce(lambda x, y: (ord(y) - ord('a') + 1) + (x * 26), s.lower(), 0)


# ============================================================================

# 172. Factorial Trailing Zeroes
# Difficulty: Easy
# link: https://leetcode.com/problems/factorial-trailing-zeroes/
# Companies: Microsoft
# Categories: Math

# ----------------------------------------------------------------------------

class Solution(object):
    def trailingZeroes(self, n):
        """
        :type n: int
        :rtype: int
        """
        # factors of 5 determins number of zeros
        # ..5..10..15..20..25..30
        # ..5...5...5..5..5*5...5
        # i = 1: n/5 -> 6
        # ...1...2...3..4....5...6
        # i = 2: n/5 -> 1
        # ...1...2...3..4....1...6

        res = 0
        while n:
            res += n / 5
            n /= 5
        return res


# ============================================================================

# 173. Binary Search Tree Iterator
# Difficulty: Medium
# link: https://leetcode.com/problems/binary-search-tree-iterator/
# Companies: Qualtrics,eBay,Google,Facebook,Bloomberg,Microsoft
# Categories: Stack,Tree,Design

# ----------------------------------------------------------------------------

# Definition for a  binary tree node
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class BSTIterator(object):
    def __init__(self, root):
        """
        :type root: TreeNode
        """
        cur = root
        self.stack = []
        while cur:
            self.stack.append(cur)
            cur = cur.left

    def hasNext(self):
        """
        :rtype: bool
        """
        return bool(self.stack)

    def next(self):
        cur = ret = self.stack.pop()
        cur = cur.right
        while cur:
            self.stack.append(cur)
            cur = cur.left
        return ret.val



# Your BSTIterator will be called like this:
# i, v = BSTIterator(root), []
# while i.hasNext(): v.append(i.next())


# ============================================================================

# 179. Largest Number
# Difficulty: Medium
# link: https://leetcode.com/problems/largest-number/
# Companies: Amazon,Microsoft
# Categories: Sort

# ----------------------------------------------------------------------------

class Solution:
    # @param {integer[]} nums
    # @return {string}
    def largestNumber(self, nums):
        return str(int(''.join(sorted(map(lambda x: str(x), nums), reverse=True, cmp=lambda x, y : 1 if str(x) + str(y) > str(y) + str(x) else -1))))


# ============================================================================

# 187. Repeated DNA Sequences
# Difficulty: Medium
# link: https://leetcode.com/problems/repeated-dna-sequences/
# Companies: Google,LinkedIn
# Categories: Hash Table,Bit Manipulation

# ----------------------------------------------------------------------------

class Solution(object):
    def findRepeatedDnaSequences(self, s):
        """
        :type s: str
        :rtype: List[str]
        """
        visited = set()
        res = []
        for i in range(0, len(s) - 10 + 1):
            sub_s = s[i: i + 10]
            if sub_s in visited and sub_s not in res:
                res.append(sub_s)
            visited.add(sub_s)
        return res


# ============================================================================

# 189. Rotate Array
# Difficulty: Easy
# link: https://leetcode.com/problems/rotate-array/
# Companies: Facebook,Microsoft,Goldman Sachs
# Categories: Array

# ----------------------------------------------------------------------------

class Solution(object):
    def rotate(self, nums, k):
        def reverse(i, j):
            while i < j:
                nums[i], nums[j] = nums[j], nums[i]
                i += 1
                j -= 1
        k %= len(nums)
        reverse(len(nums) - k, len(nums) - 1)
        reverse(0, len(nums) - k - 1)
        reverse(0, len(nums) - 1)


# ============================================================================

# 190. Reverse Bits
# Difficulty: Easy
# link: https://leetcode.com/problems/reverse-bits/
# Companies: Amazon,Apple
# Categories: Bit Manipulation

# ----------------------------------------------------------------------------

class Solution:
    # @param n, an integer
    # @return an integer
    def reverseBits(self, n):
        res = 0
        for i in xrange(32):
            n, mod = divmod(n, 2)
            res = (res << 1) | mod
        return res


# ============================================================================

# 191. Number of 1 Bits
# Difficulty: Easy
# link: https://leetcode.com/problems/number-of-1-bits/
# Companies: Box,Apple
# Categories: Bit Manipulation

# ----------------------------------------------------------------------------

class Solution(object):
    def hammingWeight(self, n):
        """
        :type n: int
        :rtype: int
        """
        count = 0
        while n:
            count += 1 & n
            n = n >> 1
        return count


# ============================================================================

# 198. House Robber
# Difficulty: Easy
# link: https://leetcode.com/problems/house-robber/
# Companies: Uber,Amazon,Google
# Categories: Dynamic Programming

# ----------------------------------------------------------------------------

class Solution(object):
    def rob(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """

        for i in range(1, len(nums)):
            nums[i] = max([nums[i - 1], nums[i]] if i - 2 < 0 else [nums[i - 1], nums[i - 2] + nums[i]])
        return nums[-1] if nums else 0


# ============================================================================

# 199. Binary Tree Right Side View
# Difficulty: Medium
# link: https://leetcode.com/problems/binary-tree-right-side-view/
# Companies: Amazon,Facebook,ByteDance,Bloomberg
# Categories: Tree,Depth-first Search,Breadth-first Search

# ----------------------------------------------------------------------------

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def rightSideView(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        if not root: return []
        bfs = [root]
        res = []
        while bfs:
            res.append(bfs[-1].val)
            bfs = [kid for node in bfs for kid in (node.left, node.right) if kid]
        return res


# ============================================================================

# 200. Number of Islands
# Difficulty: Medium
# link: https://leetcode.com/problems/number-of-islands/
# Companies: Uber,Google,Adobe,Apple,Affirm,JPMorgan,Atlassian,LinkedIn,Lyft,Amazon,Cruise Automation,Walmart Labs,Snapchat,Bloomberg,Expedia,Oracle,Facebook,Twitch,Microsoft,Qualtrics
# Categories: Depth-first Search,Breadth-first Search,Union Find

# ----------------------------------------------------------------------------

class Solution(object):
    def numIslands(self, grid):
        """
        :type grid: List[List[str]]
        :rtype: int
        """
        if not grid or not grid[0]: return 0
        def convert(from_sym, to_sym, i, j):
            def get_adj(i, j):
                return [(x, y) for x, y in [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]
                        if 0 <= x < len(grid) and 0 <= y < len(grid[0])]
            if grid[i][j] == from_sym:
                grid[i][j] = to_sym
                for adj in get_adj(i, j): convert(from_sym, to_sym, adj[0], adj[1])
        res = 0
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == '1':
                    res += 1
                    convert('1', None, i, j)
        return res


# ============================================================================

# 201. Bitwise AND of Numbers Range
# Difficulty: Medium
# link: https://leetcode.com/problems/bitwise-and-of-numbers-range/
# Companies:
# Categories: Bit Manipulation

# ----------------------------------------------------------------------------

class Solution(object):
    def rangeBitwiseAnd(self, m, n):
        """
        :type m: int
        :type n: int
        :rtype: int
        """
        res = ~0
        while ((m & res) != (n & res)):
            res = res << 1
        return res & m


# ============================================================================

# 202. Happy Number
# Difficulty: Easy
# link: https://leetcode.com/problems/happy-number/
# Companies: Google,Evernote,Apple,JPMorgan,Bloomberg,Microsoft
# Categories: Hash Table,Math

# ----------------------------------------------------------------------------

class Solution(object):
    def isHappy(self, n):
        """
        :type n: int
        :rtype: bool
        """
        visited = set()
        while n != 1:
            if n in visited: return False
            visited.add(n)
            x, n = n, 0
            while x:
                n += (x % 10) ** 2
                x /= 10

        return True


# ============================================================================

# 203. Remove Linked List Elements
# Difficulty: Easy
# link: https://leetcode.com/problems/remove-linked-list-elements/
# Companies: Pure Storage,Apple,Bloomberg
# Categories: Linked List

# ----------------------------------------------------------------------------

# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def removeElements(self, head, val):
        """
        :type head: ListNode
        :type val: int
        :rtype: ListNode
        """
        res = res_cur = ListNode('dummy')
        cur = head
        while cur:
            if cur.val == val:
                cur = cur.next
            else:
                res_cur.next = cur
                cur = cur.next
                res_cur = res_cur.next
                res_cur.next = None
        return res.next


# ============================================================================

# 204. Count Primes
# Difficulty: Easy
# link: https://leetcode.com/problems/count-primes/
# Companies: Amazon,Capital One
# Categories: Hash Table,Math

# ----------------------------------------------------------------------------

class Solution(object):
    def countPrimes(self, n):
        """
        :type n: int
        :rtype: int
        """
        if n <= 2: return 0
        primes = [True] * (n)
        primes[0] = primes[1] = False
        for i in xrange(2, n):
            if primes[i]:
                for j in xrange(i, n, i):
                    if j != i: primes[j] = False
        return primes.count(True)


# ============================================================================

# 205. Isomorphic Strings
# Difficulty: Easy
# link: https://leetcode.com/problems/isomorphic-strings/
# Companies: Amazon,Google,LinkedIn
# Categories: Hash Table

# ----------------------------------------------------------------------------

class Solution(object):
    def isIsomorphic(self, s, t):
        # ord(s_char) -> ord(t_char), (ord(t_char) << 10) -> ord(s_char)
        s_t = {}
        for i in xrange(len(s)):
            s_t[ord(s[i])] = ord(t[i])
            s_t[ord(t[i]) << 10] = ord(s[i])
        return next((False for i in xrange(len(s))
                    if not(s_t[ord(s[i])] == ord(t[i]) and s_t[ord(t[i]) << 10] == ord(s[i]))), True)


# ============================================================================

# 206. Reverse Linked List
# Difficulty: Easy
# link: https://leetcode.com/problems/reverse-linked-list/
# Companies: Google,Adobe,Apple,Mathworks,Yahoo,Walmart Labs,eBay,Amazon,Facebook,Yandex,Oracle,Cisco,Microsoft
# Categories: Linked List

# ----------------------------------------------------------------------------

# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def reverseList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        cur = head
        prev = None
        while cur:
            next_node = cur.next
            cur.next = prev
            prev, cur = cur, next_node
        return prev


# ============================================================================

# 207. Course Schedule
# Difficulty: Medium
# link: https://leetcode.com/problems/course-schedule/
# Companies: Uber,Amazon,Facebook,Apple
# Categories: Depth-first Search,Breadth-first Search,Graph,Topological Sort

# ----------------------------------------------------------------------------

class Solution(object):
    def canFinish(self, numCourses, prerequisites):
        """
        :type numCourses: int
        :type prerequisites: List[List[int]]
        :rtype: bool
        """
        graph = [[] for _ in range(numCourses)]
        flow = [0] * numCourses
        bfs = set(range(numCourses))
        for course, preq in prerequisites:
            graph[preq].append(course)
            flow[course] += 1
            if course in bfs:
                bfs.remove(course)
        flow = map(lambda x: x if x else 1, flow)
        bfs = list(bfs)
        bfs2 = []
        while bfs:
            while bfs:
                next_node = bfs.pop()
                flow[next_node] -= 1
                if flow[next_node] == 0:
                    bfs2.extend(graph[next_node])
            bfs, bfs2 = bfs2, []
        return not any(flow)


# ============================================================================

# 208. Implement Trie (Prefix Tree)
# Difficulty: Medium
# link: https://leetcode.com/problems/implement-trie-prefix-tree/
# Companies: Google,eBay,Amazon,Walmart Labs,Facebook,Microsoft
# Categories: Design,Trie

# ----------------------------------------------------------------------------

class Trie(object):

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.trie_tree = {}

    def insert(self, word):
        """
        Inserts a word into the trie.
        :type word: str
        :rtype: void
        """
        cur = self.trie_tree
        for char in word: cur = cur.setdefault(char, {})
        cur[True] = True

    def search(self, word):
        """
        Returns if the word is in the trie.
        :type word: str
        :rtype: bool
        """
        cur = self.trie_tree
        for char in word:
            if char not in cur: return False
            cur = cur[char]
        return True in cur


    def startsWith(self, prefix):
        """
        Returns if there is any word in the trie that starts with the given prefix.
        :type prefix: str
        :rtype: bool
        """
        cur = self.trie_tree
        for char in prefix:
            if char not in cur: return False
            cur = cur[char]
        return True


# Your Trie object will be instantiated and called as such:
# obj = Trie()
# obj.insert(word)
# param_2 = obj.search(word)
# param_3 = obj.startsWith(prefix)


# ============================================================================

# 209. Minimum Size Subarray Sum
# Difficulty: Medium
# link: https://leetcode.com/problems/minimum-size-subarray-sum/
# Companies: Amazon,Google,Bloomberg,Goldman Sachs
# Categories: Array,Two Pointers,Binary Search

# ----------------------------------------------------------------------------

class Solution(object):
    def minSubArrayLen(self, s, nums):
        """
        :type s: int
        :type nums: List[int]
        :rtype: int
        """
        min_len = total = start = 0
        for end, num in enumerate(nums):
            total += num
            while total >= s:
                min_len = min(end - start + 1, min_len or float('inf'))
                total -= nums[start]
                start += 1
        return min_len


# ============================================================================

# 210. Course Schedule II
# Difficulty: Medium
# link: https://leetcode.com/problems/course-schedule-ii/
# Companies: Uber,Google,Intuit,Amazon,Facebook,DoorDash,Microsoft
# Categories: Depth-first Search,Breadth-first Search,Graph,Topological Sort

# ----------------------------------------------------------------------------

class Solution(object):
    def findOrder(self, numCourses, prerequisites):
        """
        :type numCourses: int
        :type prerequisites: List[List[int]]
        :rtype: List[int]
        """
        graph = [set() for _ in xrange(numCourses)]
        flow_in = [0 for _ in xrange(numCourses)]
        for course, prereq in prerequisites:
            if course not in graph[prereq]:
                graph[prereq].add(course)
                flow_in[course] += 1
        bfs = [node for node, in_count in enumerate(flow_in) if in_count == 0]
        for node in bfs: flow_in[node] = 1
        res = []
        while bfs:
            adjs = []
            for node in bfs:
                flow_in[node] -= 1
                if not flow_in[node]:
                    res.append(node)
                    for to_node in graph[node]: adjs.append(to_node)
            bfs = adjs
        return res if len(res) == numCourses else []


# ============================================================================

# 211. Add and Search Word - Data structure design
# Difficulty: Medium
# link: https://leetcode.com/problems/add-and-search-word-data-structure-design/
# Companies: Amazon,Facebook,Microsoft
# Categories: Backtracking,Design,Trie

# ----------------------------------------------------------------------------

class WordDictionary(object):

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.trie_tree = {}

    def addWord(self, word):
        """
        Adds a word into the data structure.
        :type word: str
        :rtype: void
        """
        cur = self.trie_tree
        for char in word:
            cur = cur.setdefault(char, {})
        cur[True] = word

    def search(self, word):
        """
        Returns if the word is in the data structure. A word could contain the dot character '.' to represent any one letter.
        :type word: str
        :rtype: bool
        """
        stack = [self.trie_tree]
        for char in word:
            if not stack: return False
            if char == '.':
                stack = [cur[cur_char] for cur in stack for cur_char in cur if cur_char != True]
            else:
                stack = [cur[char] for cur in stack if char in cur]
        return any(cur[True] for cur in stack if True in cur)


# Your WordDictionary object will be instantiated and called as such:
# obj = WordDictionary()
# obj.addWord(word)
# param_2 = obj.search(word)


# ============================================================================

# 212. Word Search II
# Difficulty: Hard
# link: https://leetcode.com/problems/word-search-ii/
# Companies: Uber,Google,Apple,Amazon,Facebook,Microsoft
# Categories: Backtracking,Trie

# ----------------------------------------------------------------------------

class Solution(object):
    def findWords(self, board, words):
        """
        :type board: List[List[str]]
        :type words: List[str]
        :rtype: List[str]
        """
        m, n = len(board), len(board[0])
        tree = {}
        for word in words:
            cur = tree
            for char in word:
                cur = cur.setdefault(char, {})
            cur[True] = True
        seen = set()
        word = []
        def dfs(i, j, cur):
            word.append(board[i][j])
            if board[i][j] in cur:
                if True in cur[board[i][j]]: seen.add(''.join(word))
                tmp, board[i][j] = board[i][j], None

                for x, y in [(i + dx, j + dy)
                             for dx, dy in zip([1,0,-1,0], [0,1,0,-1])
                             if 0 <= i + dx < m and 0 <= j + dy < n]:
                    if board[x][y]:
                        dfs(x, y, cur[tmp])
                board[i][j] = tmp
            word.pop()
        for i in range(m):
            for j in range(n):
                dfs(i, j, tree)
        return list(seen)


# ============================================================================

# 213. House Robber II
# Difficulty: Medium
# link: https://leetcode.com/problems/house-robber-ii/
# Companies: Google
# Categories: Dynamic Programming

# ----------------------------------------------------------------------------

class Solution(object):
    def rob(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        def _rob(nums):
            for i in range(1, len(nums)):
                nums[i] = max([nums[i - 1], nums[i]] if i - 2 < 0 else [nums[i - 1], nums[i - 2] + nums[i]])
            return nums[-1] if nums else 0
        return max(_rob(nums[:-1]), _rob(nums[1:])) if len(nums) > 1 else (nums or [0]) [0]


# ============================================================================

# 215. Kth Largest Element in an Array
# Difficulty: Medium
# link: https://leetcode.com/problems/kth-largest-element-in-an-array/
# Companies: Amazon,Google,Facebook,Yahoo,LinkedIn
# Categories: Divide and Conquer,Heap

# ----------------------------------------------------------------------------

class Solution(object):
    def findKthLargest(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        import heapq
        if len(nums) < k: return
        for i, num in enumerate(nums): nums[i] = -num
        heapq.heapify(nums)
        cur = None
        for _ in range(k): cur = -heapq.heappop(nums)
        return cur


# ============================================================================

# 216. Combination Sum III
# Difficulty: Medium
# link: https://leetcode.com/problems/combination-sum-iii/
# Companies: Microsoft
# Categories: Array,Backtracking

# ----------------------------------------------------------------------------

class Solution(object):
    def combinationSum3(self, k, n):
        def _comb(idx, k, n, cur=[], res=[]):
            if n == 0 and k == 0:
                res.append(cur[:])
                return res
            elif idx > 9 or k == 0 or n < 0: return res
            else:
                cur.append(idx)
                _comb(idx + 1, k - 1, n - idx)
                cur.pop()
                _comb(idx + 1, k, n)
                return res
        return _comb(1, k, n)


# ============================================================================

# 217. Contains Duplicate
# Difficulty: Easy
# link: https://leetcode.com/problems/contains-duplicate/
# Companies: Oracle,Amazon,Microsoft
# Categories: Array,Hash Table

# ----------------------------------------------------------------------------

class Solution(object):
    def containsDuplicate(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        return len(nums) != len(set(nums))


# ============================================================================

# 219. Contains Duplicate II
# Difficulty: Easy
# link: https://leetcode.com/problems/contains-duplicate-ii/
# Companies: Airbnb,Google,Adobe,Palantir Technologies
# Categories: Array,Hash Table

# ----------------------------------------------------------------------------

class Solution(object):
    def containsNearbyDuplicate(self, nums, k):
        last_seen = {}
        for i, num in enumerate(nums):
            if i - last_seen.get(num, float('-inf')) <= k:
                return True
            last_seen[num] = i
        return False


# ============================================================================

# 221. Maximal Square
# Difficulty: Medium
# link: https://leetcode.com/problems/maximal-square/
# Companies: Uber,Google,Huawei,Amazon,Oracle,VMware,Citadel
# Categories: Dynamic Programming

# ----------------------------------------------------------------------------

class Solution(object):
    def maximalSquare(self, matrix):
        """
        :type matrix: List[List[str]]
        :rtype: int
        """
        from itertools import chain
        m, n = len(matrix), len(matrix[0]) if matrix else 0
        matrix = [map(int, row) for row in matrix]
        # check if the first row or colum contains a 1
        max_w = any(chain((matrix[0] if matrix else []), (row[0] for row in matrix)))
        for i in range(1, m):
            for j in range(1, n):
                min_wh = min(matrix[i - 1][j], matrix[i][j - 1])
                is_inc = matrix[i - min_wh][j - min_wh] and matrix[i][j]
                matrix[i][j] = (min_wh if matrix[i][j] else 0) + is_inc
                max_w = max(max_w, matrix[i][j])
        return max_w ** 2


# ============================================================================

# 222. Count Complete Tree Nodes
# Difficulty: Medium
# link: https://leetcode.com/problems/count-complete-tree-nodes/
# Companies: Amazon,Google
# Categories: Binary Search,Tree

# ----------------------------------------------------------------------------

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def countNodes(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        def _countNodes(node):
            left_node, left_depth = node, 0
            while left_node: left_node, left_depth = left_node.left, left_depth + 1
            right_node, right_depth = node, 0
            while right_node: right_node, right_depth = right_node.right, right_depth + 1
            if left_depth == right_depth: return 2 ** left_depth - 1
            else: return _countNodes(node.left) + 1 + _countNodes(node.right)
        return _countNodes(root)


# ============================================================================

# 223. Rectangle Area
# Difficulty: Medium
# link: https://leetcode.com/problems/rectangle-area/
# Companies: Facebook,Apple
# Categories: Math

# ----------------------------------------------------------------------------

class Solution(object):
    def computeArea(self, A, B, C, D, E, F, G, H):
        """
        :type A: int
        :type B: int
        :type C: int
        :type D: int
        :type E: int
        :type F: int
        :type G: int
        :type H: int
        :rtype: int
        """
        width, height = (min(C, G) - max(A, E)), (min(D, H) - max(B, F))
        overlap = width * height if width > 0 and height > 0 else 0
        area1 = (C - A) * (D - B)
        area2 = (G - E) * (H - F)
        return area1 + area2 - overlap


# ============================================================================

# 224. Basic Calculator
# Difficulty: Hard
# link: https://leetcode.com/problems/basic-calculator/
# Companies: Uber,Roblox,Robinhood,Snapchat,Amazon,Facebook,Microsoft
# Categories: Math,Stack

# ----------------------------------------------------------------------------

class Solution(object):
    def calculate(self, s):
        """
        :type s: str
        :rtype: int
        """
        s = list(('((%s)'%s).replace(' ', ''))
        nested_brackets, num = [], []
        for i, c in enumerate(s):
            if c == '(':
                nested_brackets.append([0, '+'])
                num = []
            elif c == '-' and s[i-1] in '+-(':
                prev_sign = nested_brackets[-1][1]
                nested_brackets[-1][1] = '+' if prev_sign == '-' else '+'
            elif c in '+-)':
                if num:
                    num = int(''.join(num))
                    sign = (1 if nested_brackets[-1][1] == '+' else -1)
                    nested_brackets[-1][0] += sign * num
                    num = []
                    if c == ')':
                        num = list(str(nested_brackets[-1][0]))
                        nested_brackets.pop()
                    else: nested_brackets[-1][1] = c
            elif c.isdigit(): num.append(c)
        return int(''.join(num))


# ============================================================================

# 225. Implement Stack using Queues
# Difficulty: Easy
# link: https://leetcode.com/problems/implement-stack-using-queues/
# Companies: Amazon,Microsoft
# Categories: Stack,Design

# ----------------------------------------------------------------------------

class MyStack(object):

    def __init__(self):
        import Queue
        self.pri, self.sec = Queue.PriorityQueue(), Queue.PriorityQueue()

    def push(self, x):
        self.pri.put(x)

    def pop(self):
        while True:
            tmp = self.pri.get() if not self.pri.empty() else None
            if not self.pri.empty(): self.sec.put(tmp)
            else:
                self.pri, self.sec = self.sec, self.pri
                return tmp


    def top(self):
        while True:
            tmp = self.pri.get() if not self.pri.empty() else None
            if tmp is not None: self.sec.put(tmp)
            if self.pri.empty():
                self.pri, self.sec = self.sec, self.pri
                return tmp

    def empty(self):
        return self.pri.empty()


# Your MyStack object will be instantiated and called as such:
# obj = MyStack()https://leetcode.com/submissions/detail/150386250/
# obj.push(x)
# param_2 = obj.pop()
# param_3 = obj.top()
# param_4 = obj.empty()


# ============================================================================

# 226. Invert Binary Tree
# Difficulty: Easy
# link: https://leetcode.com/problems/invert-binary-tree/
# Companies: Amazon,Google
# Categories: Tree

# ----------------------------------------------------------------------------

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def invertTree(self, root):
        """
        :type root: TreeNode
        :rtype: TreeNode
        """
        if not root: return
        bfs = [root]
        while bfs:
            for node in bfs:
                node.left, node.right = node.right, node.left
            bfs = [kid for node in bfs for kid in [node.left, node.right] if kid]
        return root


# ============================================================================

# 227. Basic Calculator II
# Difficulty: Medium
# link: https://leetcode.com/problems/basic-calculator-ii/
# Companies: Uber,Houzz,Reddit,Apple,Amazon,Facebook,Microsoft
# Categories: String

# ----------------------------------------------------------------------------

class Solution(object):
    def calculate(self, s):
        """
        :type s: str
        :rtype: int
        """
        import re, operator as op
        expr = map(lambda x: int(x) if x.isdigit() else x, re.split('(\*|/|\+|-)', s.replace(' ', '')))
        def eval_expr(ops):
            op_lambda = {'+': op.add, '-': op.sub, '*': op.mul, '/': op.div}
            new_expr = []
            for el in expr:
                if new_expr and type(new_expr[-1]) != int and new_expr[-1] in ops:
                    new_expr[-2] = op_lambda[new_expr[-1]](new_expr[-2], el)
                    new_expr.pop()
                else: new_expr.append(el)
            return new_expr
        expr = eval_expr('*/')
        expr = eval_expr('+-')
        return expr[0]


# ============================================================================

# 228. Summary Ranges
# Difficulty: Medium
# link: https://leetcode.com/problems/summary-ranges/
# Companies: Amazon,Facebook,Capital One
# Categories: Array

# ----------------------------------------------------------------------------

class Solution(object):
    def summaryRanges(self, nums):
        """
        :type nums: List[int]
        :rtype: List[str]
        """
        ranges = []
        for i in range(len(nums)):
            if not ranges or ranges[-1][1] != nums[i] - 1:
                ranges.append([nums[i], nums[i]])
            else:
                ranges[-1][1] = nums[i]
        return [str(i) + '->' + str(j) if i != j else str(i) for i, j in ranges]


# ============================================================================

# 229. Majority Element II
# Difficulty: Medium
# link: https://leetcode.com/problems/majority-element-ii/
# Companies: Uber,Amazon,Google,Microsoft,Bloomberg
# Categories: Array

# ----------------------------------------------------------------------------

class Solution(object):
    def majorityElement(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        if not nums or len(nums) <= 1:
            return nums[:]

        candidate1, candidate2, counter1, counter2 = float('inf'), float('inf'), 0, 0
        for num in nums:
            if num == candidate1:
                counter1 += 1
            elif num == candidate2:
                counter2 += 1
            elif counter1 == 0:
                candidate1, counter1 = num, 1
            elif counter2 == 0:
                candidate2, counter2 = num, 1
            else:
                counter1 -= 1
                counter2 -= 1
        return [ i for i in [candidate1, candidate2] if nums.count(i) > (len(nums) / 3.)]


# ============================================================================

# 230. Kth Smallest Element in a BST
# Difficulty: Medium
# link: https://leetcode.com/problems/kth-smallest-element-in-a-bst/
# Companies: Uber,Facebook,Amazon,TripleByte,Microsoft
# Categories: Binary Search,Tree

# ----------------------------------------------------------------------------

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def kthSmallest(self, root, k):
        """
        :type root: TreeNode
        :type k: int
        :rtype: int
        """
        stack = []
        def move_left(node):
            while node:
                stack.append(node)
                node = node.left
        move_left(root)
        for i in xrange(k):
            next_elem = stack.pop()
            if next_elem.right:
                move_left(next_elem.right)
        return next_elem.val


# ============================================================================

# 231. Power of Two
# Difficulty: Easy
# link: https://leetcode.com/problems/power-of-two/
# Companies: Google,Apple
# Categories: Math,Bit Manipulation

# ----------------------------------------------------------------------------

class Solution(object):
    def isPowerOfTwo(self, n):
        return (n > 0) and (n & (n - 1)) == 0


# ============================================================================

# 232. Implement Queue using Stacks
# Difficulty: Easy
# link: https://leetcode.com/problems/implement-queue-using-stacks/
# Companies: Oracle,Mathworks,Microsoft,Bloomberg
# Categories: Stack,Design

# ----------------------------------------------------------------------------

class MyQueue(object):
    def __init__(self):
        self.incoming = []
        self.outgoing = []

    def push(self, x): self.incoming.append(x)

    def _move_to_outgoing(self):
        if not self.outgoing:
            while self.incoming:
                self.outgoing.append(self.incoming.pop())
        return self.outgoing

    def pop(self): return self._move_to_outgoing().pop()

    def peek(self): return self._move_to_outgoing()[-1]

    def empty(self): return not (self.incoming or self.outgoing)


# ============================================================================

# 234. Palindrome Linked List
# Difficulty: Easy
# link: https://leetcode.com/problems/palindrome-linked-list/
# Companies: Amazon,Adobe,Microsoft,IXL,Apple
# Categories: Linked List,Two Pointers

# ----------------------------------------------------------------------------

# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def isPalindrome(self, head):
        """
        :type head: ListNode
        :rtype: bool
        """
        if not head or not head.next: return True

        num_elem = 0
        cur = head
        while cur:
            num_elem += 1
            cur = cur.next
        cur = head
        prev = None
        mid = num_elem / 2
        for _ in range(mid):
            next_node = cur.next
            cur.next = prev
            prev, cur = cur, next_node
        if num_elem & 1: cur = cur.next
        left, right = prev, cur
        while left or right:
            if left.val != right.val: return False
            left, right = left.next, right.next
        return True


# ============================================================================

# 235. Lowest Common Ancestor of a Binary Search Tree
# Difficulty: Easy
# link: https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-search-tree/
# Companies: Amazon,Facebook,LinkedIn
# Categories: Tree

# ----------------------------------------------------------------------------

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def lowestCommonAncestor(self, root, p, q):
        """
        :type root: TreeNode
        :type p: TreeNode
        :type q: TreeNode
        :rtype: TreeNode
        """
        def _lowestCommonAncestor(node):
            if not node or node == p or node == q: return node
            left, right = _lowestCommonAncestor(node.left), _lowestCommonAncestor(node.right)
            return node if (left and right) else (left or right)
        return _lowestCommonAncestor(root)


# ============================================================================

# 236. Lowest Common Ancestor of a Binary Tree
# Difficulty: Medium
# link: https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree/
# Companies: Google,ByteDance,Apple,LinkedIn,Amazon,Visa,Facebook,Bloomberg,Oracle,Microsoft,Zillow
# Categories: Tree

# ----------------------------------------------------------------------------

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def lowestCommonAncestor(self, root, p, q):
        def _searchAncestor(node):
            if node in [p, q, None]: return node
            else:
                l, r = _searchAncestor(node.left), _searchAncestor(node.right)
                return node if (l and r) else (l or r)
        return _searchAncestor(root)


# ============================================================================

# 237. Delete Node in a Linked List
# Difficulty: Easy
# link: https://leetcode.com/problems/delete-node-in-a-linked-list/
# Companies: Microsoft
# Categories: Linked List

# ----------------------------------------------------------------------------

# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def deleteNode(self, node):
        """
        :type node: ListNode
        :rtype: void Do not return anything, modify node in-place instead.
        """
        node.val = node.next.val
        node.next = node.next.next


# ============================================================================

# 238. Product of Array Except Self
# Difficulty: Medium
# link: https://leetcode.com/problems/product-of-array-except-self/
# Companies: Lyft,Evernote,Apple,Tableau,Goldman Sachs,Amazon,Asana,Facebook,Oracle,Microsoft,VMware,Google
# Categories: Array

# ----------------------------------------------------------------------------

class Solution(object):
    def productExceptSelf(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        dp = [1]
        for num in reversed(nums): dp.append(num * dp[-1])
        dp = dp[::-1]
        mul_so_far = 1
        res = []
        for i in range(len(nums)):
            res.append(mul_so_far * dp[i + 1])
            mul_so_far *= nums[i]
        return res


# ============================================================================

# 240. Search a 2D Matrix II
# Difficulty: Medium
# link: https://leetcode.com/problems/search-a-2d-matrix-ii/
# Companies: Salesforce,Paypal,Amazon,Facebook,SAP,Microsoft,Citadel
# Categories: Binary Search,Divide and Conquer

# ----------------------------------------------------------------------------

class Solution(object):
    def searchMatrix(self, matrix, target):
        """
        :type matrix: List[List[int]]
        :type target: int
        :rtype: bool
        """
        if not matrix or not any(matrix):
            return False
        col_i = len(matrix[0]) - 1
        for row in matrix:
            while row[col_i] > target and col_i > 0:
                col_i -= 1
            if row[col_i] == target:
                return True
        return False


# ============================================================================

# 241. Different Ways to Add Parentheses
# Difficulty: Medium
# link: https://leetcode.com/problems/different-ways-to-add-parentheses/
# Companies: Google,Facebook
# Categories: Divide and Conquer

# ----------------------------------------------------------------------------

import re
class Solution(object):
    def diffWaysToCompute(self, input_vals):
        """
        :type input: str
        :rtype: List[int]
        """
        list_sep_vals = []
        buf = []
        list_sep_vals = re.split('([^\d])', input_vals)
        res = []
        ops = { '+': operator.add, '-': operator.sub, '*': operator.mul, '/': operator.div }
        for i in range(0, len(list_sep_vals), 2):
            list_sep_vals[i] = int(list_sep_vals[i])
        dp = {}
        def _diffWaysToCompute(start, end):
            if start == end - 1:
                return [list_sep_vals[start]]
            key = '%d_%d'%(start, end)
            if key in dp:
                return dp[key]
            dp[key] = []
            for i in range(start + 1, end, 2):
                left_combo = _diffWaysToCompute(start, i)
                right_combo = _diffWaysToCompute(i + 1, end)
                for l in left_combo:
                    for r in right_combo:
                        dp[key].append(ops[list_sep_vals[i]](l, r))
            return dp[key]



        return _diffWaysToCompute(0, len(list_sep_vals))


# ============================================================================

# 242. Valid Anagram
# Difficulty: Easy
# link: https://leetcode.com/problems/valid-anagram/
# Companies: Amazon,Facebook,Microsoft,Bloomberg
# Categories: Hash Table,Sort

# ----------------------------------------------------------------------------

class Solution(object):
    def isAnagram(self, s, t):
        from collections import Counter
        s_cnt, t_cnt = Counter(s), Counter(t)
        return len(s) == len(t) and all(s_cnt[char] == t_cnt[char] for char in s)


# ============================================================================

# 243. Shortest Word Distance
# Difficulty: Easy
# link: https://leetcode.com/problems/shortest-word-distance/
# Companies: Paypal,LinkedIn
# Categories: Array

# ----------------------------------------------------------------------------

class Solution(object):
    def shortestDistance(self, words, word1, word2):
        last_seen = {}
        min_dist = float('inf')
        for j, word in enumerate(words):
            if word1 in last_seen and word == word2:
                min_dist = min(min_dist, j - last_seen[word1])
            elif word2 in last_seen and word == word1:
                min_dist = min(min_dist, j - last_seen[word2])
            last_seen[word] = j
        return min_dist


# ============================================================================

# 246. Strobogrammatic Number
# Difficulty: Easy
# link: https://leetcode.com/problems/strobogrammatic-number/
# Companies: Google,Facebook
# Categories: Hash Table,Math

# ----------------------------------------------------------------------------

class Solution(object):
    def isStrobogrammatic(self, num):
        return set('16890') >= set(num) and \
                num == num[::-1].replace('6','.').replace('9','6').replace('.','9')


# ============================================================================

# 250. Count Univalue Subtrees
# Difficulty: Medium
# link: https://leetcode.com/problems/count-univalue-subtrees/
# Companies: Box
# Categories: Tree

# ----------------------------------------------------------------------------

class Solution(object):
    def countUnivalSubtrees(self, root):
        self.cnt = 0
        def _cnt(node, par):
            if not node: return True

            l, r = _cnt(node.left, node), _cnt(node.right, node)

            self.cnt += l and r
            return l and r and node.val == par.val
        _cnt(root, root)
        return self.cnt


# ============================================================================

# 252. Meeting Rooms
# Difficulty: Easy
# link: https://leetcode.com/problems/meeting-rooms/
# Companies: Amazon,Google,Bloomberg
# Categories: Sort

# ----------------------------------------------------------------------------

class Solution(object):
    def canAttendMeetings(self, intervals):
        intervals.sort()
        return all(intervals[i][1] <= intervals[i + 1][0] for i in range(len(intervals) - 1))


# ============================================================================

# 253. Meeting Rooms II
# Difficulty: Medium
# link: https://leetcode.com/problems/meeting-rooms-ii/
# Companies: Uber,Google,Apple,Yelp,Booking.com,Quora,Amazon,Lyft,Facebook,Bloomberg,Oracle,Microsoft
# Categories: Heap,Greedy,Sort

# ----------------------------------------------------------------------------

class Solution(object):
    def minMeetingRooms(self, intervals):
        """
        :type intervals: List[List[int]]
        :rtype: int
        """
        events = sorted(event
                   for interval in intervals
                   for event in [(interval[0], 1), (interval[1], -1)])
        overlap = 0
        cur = 0
        for e in events:
            cur += e[1]
            overlap = max(overlap, cur)
        return overlap


# ============================================================================

# 257. Binary Tree Paths
# Difficulty: Easy
# link: https://leetcode.com/problems/binary-tree-paths/
# Companies: Amazon,Facebook
# Categories: Tree,Depth-first Search

# ----------------------------------------------------------------------------

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def binaryTreePaths(self, root):
        """
        :type root: TreeNode
        :rtype: List[str]
        """
        if not root: return []
        bfs = [(root, str(root.val))]
        paths = []
        while bfs:
            paths.extend([path for node, path in bfs if not node.left and not node.right])
            bfs = [(kid, "%s->%s" %(path, str(kid.val))) for node, path in bfs for kid in [node.left, node.right] if kid]
        return paths


# ============================================================================

# 258. Add Digits
# Difficulty: Easy
# link: https://leetcode.com/problems/add-digits/
# Companies: Apple
# Categories: Math

# ----------------------------------------------------------------------------

class Solution(object):
    def addDigits(self, num):
        while num >= 10:
            n_num = 0
            while num:
                num, dig = divmod(num, 10)
                n_num += dig
            num = n_num
        return num


# ============================================================================

# 260. Single Number III
# Difficulty: Medium
# link: https://leetcode.com/problems/single-number-iii/
# Companies: Amazon
# Categories: Bit Manipulation

# ----------------------------------------------------------------------------

class Solution(object):
    def singleNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        xord = 0
        for num in nums:
            xord ^= num

        for i in xrange(32):
            if (1 << i) & xord:
                num1 = 0
                num2 = 0
                for num in nums:
                    if (1 << i) & num:
                        num1 ^= num
                    else:
                        num2 ^= num
                return [num1, num2]


# ============================================================================

# 261. Graph Valid Tree
# Difficulty: Medium
# link: https://leetcode.com/problems/graph-valid-tree/
# Companies: Amazon,LinkedIn
# Categories: Depth-first Search,Breadth-first Search,Union Find,Graph

# ----------------------------------------------------------------------------

class Solution(object):
    def validTree(self, n, edges):
        uf = list(range(n))

        def find_par(node):

            path = [node]
            while uf[node] != node:
                node = uf[node]
                path.append(node)

            for i in range(len(path) - 1):
                uf[path[i]] = path[-1]

            return path[-1]

        for u, v in edges:
            u, v = find_par(u), find_par(v)
            if u == v: return False
            uf[u] = v

        return sum(node == i for i, node in enumerate(uf)) == 1


# ============================================================================

# 265. Paint House II
# Difficulty: Hard
# link: https://leetcode.com/problems/paint-house-ii/
# Companies: LinkedIn
# Categories: Dynamic Programming

# ----------------------------------------------------------------------------

class Solution(object):
    def minCostII(self, costs):
        from itertools import chain
        if not costs or not costs[0]: return 0
        m, n = len(costs), len(costs[0])
        if n == 1: return costs[0][0]
        totals = [0] * n
        for row in costs:
            totals = [row[i] + min(totals[j] for j in range(n) if i != j)
                      for i in range(n)]
        return min(totals)


# ============================================================================

# 266. Palindrome Permutation
# Difficulty: Easy
# link: https://leetcode.com/problems/palindrome-permutation/
# Companies: Amazon,Google,Facebook
# Categories: Hash Table

# ----------------------------------------------------------------------------

class Solution(object):
    def canPermutePalindrome(self, s):
        from collections import Counter
        cnt = Counter(s)
        return sum((c % 2 for c in cnt.values()) or 0) < 2


# ============================================================================

# 268. Missing Number
# Difficulty: Easy
# link: https://leetcode.com/problems/missing-number/
# Companies: Amazon,Google,Microsoft,Apple
# Categories: Array,Math,Bit Manipulation

# ----------------------------------------------------------------------------

class Solution(object):
    def missingNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        n = len(nums) + 1
        return (n * (n - 1)) / 2 - sum(nums)


# ============================================================================

# 269. Alien Dictionary
# Difficulty: Hard
# link: https://leetcode.com/problems/alien-dictionary/
# Companies: Uber,Google,Pinterest,Airbnb,Amazon,Facebook,Bloomberg,Microsoft
# Categories: Graph,Topological Sort

# ----------------------------------------------------------------------------

class Solution(object):
    def alienOrder(self, words):
        from collections import defaultdict
        G = defaultdict(set)
        deg = {char: 0 for word in words for char in word}

        for i in range(len(words) - 1):
            wrd1, wrd2 = words[i], words[i + 1]
            for j in range(min(len(wrd1), len(wrd2))):
                if wrd1[j] != wrd2[j]:
                    if wrd2[j] not in G[wrd1[j]]:
                        G[wrd1[j]].add(wrd2[j])
                        deg[wrd2[j]] += 1
                    break

        bfs = [n for n, cnt in deg.iteritems() if not cnt]
        res = []

        def to_visit(node):
            deg[node] -= 1
            return not deg[node]

        while bfs:
            res += bfs
            bfs = [adj
                   for node in bfs
                   for adj in G[node]
                   if to_visit(adj)]

        return ''.join(res) if len(res) == len(deg) else ""


# ============================================================================

# 274. H-Index
# Difficulty: Medium
# link: https://leetcode.com/problems/h-index/
# Companies: TripAdvisor,Adobe
# Categories: Hash Table,Sort

# ----------------------------------------------------------------------------

class Solution(object):
    def hIndex(self, citations):
        counter = [0] * (len(citations) + 1)
        for citation in citations:
            if citation > len(citations):
                citation = len(citations)
            counter[citation] += 1
        sum_from_right = 0
        for i in range(len(citations), -1, -1):
            sum_from_right += counter[i]
            if sum_from_right >= i:
                return i


# ============================================================================

# 278. First Bad Version
# Difficulty: Easy
# link: https://leetcode.com/problems/first-bad-version/
# Companies: Facebook
# Categories: Binary Search

# ----------------------------------------------------------------------------

# The isBadVersion API is already defined for you.
# @param version, an integer
# @return a bool
# def isBadVersion(version):

class Solution(object):
    def firstBadVersion(self, n):
        """
        :type n: int
        :rtype: int
        """
        low, high = 1, n

        while low <= high:
            mid = (low + high) / 2
            if isBadVersion(mid):
                if (mid - 1) < low or not isBadVersion(mid - 1): return mid
                high = mid - 1
            else:
                low = mid + 1


# ============================================================================

# 279. Perfect Squares
# Difficulty: Medium
# link: https://leetcode.com/problems/perfect-squares/
# Companies: Amazon,Google,Cisco,Apple
# Categories: Math,Dynamic Programming,Breadth-first Search

# ----------------------------------------------------------------------------

class Solution(object):
    def numSquares(self, n):
        """
        :type n: int
        :rtype: int
        """
        bfs = [n]
        count = 0
        visited = {n}
        sq = {i: i**2 for i in range(1, int(n ** (1./2)) + 1)}
        while bfs:
            if not all(bfs): return count
            bfs = [(i - sq[j])
                      for i in bfs
                      for j in range(1, int(i ** (1./2)) + 1)
                      if (i - sq[j]) not in visited and (i - sq[j]) >= 0 and (visited.add(i - sq[j]) is None)]
            count += 1
        return 0


# ============================================================================

# 281. Zigzag Iterator
# Difficulty: Medium
# link: https://leetcode.com/problems/zigzag-iterator/
# Companies: Amazon
# Categories: Design

# ----------------------------------------------------------------------------

class ZigzagIterator(object):

    def __init__(self, v1, v2):
        self.v1, self.v2 = v1, v2
        self.i = self.j = self.tog = 0

    def next(self):
        if self.i >= len(self.v1) or self.tog and self.j < len(self.v2):
            res, self.j = self.v2[self.j], self.j + 1
        else:
            res, self.i = self.v1[self.i], self.i + 1
        self.tog = not self.tog
        return res

    def hasNext(self):
        return self.i < len(self.v1) or self.j < len(self.v2)


# ============================================================================

# 283. Move Zeroes
# Difficulty: Easy
# link: https://leetcode.com/problems/move-zeroes/
# Companies: Google,Yahoo,Goldman Sachs,Amazon,Facebook,Yandex,Bloomberg,Walmart Labs,Microsoft,Zillow
# Categories: Array,Two Pointers

# ----------------------------------------------------------------------------

class Solution(object):
    def moveZeroes(self, nums):
        """
        :type nums: List[int]
        :rtype: void Do not return anything, modify nums in-place instead.
        """
        i = 0
        for j, num in enumerate(nums):
            if num != 0:
                nums[i] = num
                i += 1
        for j in range(i, len(nums)):
            nums[j] = 0


# ============================================================================

# 284. Peeking Iterator
# Difficulty: Medium
# link: https://leetcode.com/problems/peeking-iterator/
# Companies: Amazon
# Categories: Design

# ----------------------------------------------------------------------------

# Below is the interface for Iterator, which is already defined for you.
#
# class Iterator(object):
#     def __init__(self, nums):
#         """
#         Initializes an iterator object to the beginning of a list.
#         :type nums: List[int]
#         """
#
#     def hasNext(self):
#         """
#         Returns true if the iteration has more elements.
#         :rtype: bool
#         """
#
#     def next(self):
#         """
#         Returns the next element in the iteration.
#         :rtype: int
#         """

class PeekingIterator(object):
    def __init__(self, iterator):
        """
        Initialize your data structure here.
        :type iterator: Iterator
        """
        self.iterator = iterator
        self.cache = None

    def peek(self):
        """
        Returns the next element in the iteration without advancing the iterator.
        :rtype: int
        """
        if self.cache is None and self.iterator.hasNext():
            self.cache = self.iterator.next()
        return self.cache

    def next(self):
        """
        :rtype: int
        """
        self.peek()
        cache = self.cache
        self.cache = None
        return cache

    def hasNext(self):
        """
        :rtype: bool
        """
        self.peek()
        return not bool(self.cache is None)

# Your PeekingIterator object will be instantiated and called as such:
# iter = PeekingIterator(Iterator(nums))
# while iter.hasNext():
#     val = iter.peek()   # Get the next element but not advance the iterator.
#     iter.next()         # Should return the same value as [val].


# ============================================================================

# 289. Game of Life
# Difficulty: Medium
# link: https://leetcode.com/problems/game-of-life/
# Companies: Dropbox,Amazon,Google,Reddit,Goldman Sachs
# Categories: Array

# ----------------------------------------------------------------------------

class Solution(object):
    def gameOfLife(self, board):
        if not board: return
        x_y_diff = [-1, 0, 1]
        m, n = len(board), len(board[0])
        for i in xrange(m):
            for j in xrange(n):
                neighbors = [(i + x, j + y) for x in x_y_diff for y in x_y_diff if (x or y)]

                count_life = 0
                for x, y in neighbors:
                    if (0 <= x < m) and (0 <= y < n) and (board[x][y] in [1, 2]):
                        count_life += 1

                if board[i][j] and (count_life < 2 or count_life > 3): board[i][j] = 2
                elif not board[i][j] and count_life == 3: board[i][j] = 3

        for i in xrange(m):
            for j in xrange(n):
                board[i][j] = 1 & board[i][j]


# ============================================================================

# 290. Word Pattern
# Difficulty: Easy
# link: https://leetcode.com/problems/word-pattern/
# Companies: Capital One
# Categories: Hash Table

# ----------------------------------------------------------------------------

class Solution(object):
    def wordPattern(self, pattern, str):
        """
        :type pattern: str
        :type str: str
        :rtype: bool
        """
        pat_to_word = {}
        word_to_pat = {}
        words = str.split(' ')
        if len(words) != len(pattern): return False
        for i, word in enumerate(words):
            if pattern[i] in pat_to_word and pat_to_word[pattern[i]] != word or \
                word in word_to_pat and word_to_pat[word] != pattern[i]: return False
            pat_to_word.setdefault(pattern[i], word)
            word_to_pat.setdefault(word, pattern[i])
        return True


# ============================================================================

# 292. Nim Game
# Difficulty: Easy
# link: https://leetcode.com/problems/nim-game/
# Companies: Adobe
# Categories: Brainteaser,Minimax

# ----------------------------------------------------------------------------

class Solution(object):
    def canWinNim(self, n):
        """
        :type n: int
        :rtype: bool
        """
        return  (n % 4) != 0


# ============================================================================

# 294. Flip Game II
# Difficulty: Medium
# link: https://leetcode.com/problems/flip-game-ii/
# Companies: Google
# Categories: Backtracking,Minimax

# ----------------------------------------------------------------------------

class Solution(object):
    def canWin(self, s, memo={}):
        if s in memo: return memo[s]
        return memo.setdefault(s, \
            any(s[k: k + 2] == "++" and not self.canWin(s[:k] + "--" + s[k + 2:])
                for k in range(len(s) - 1)))


# ============================================================================

# 297. Serialize and Deserialize Binary Tree
# Difficulty: Hard
# link: https://leetcode.com/problems/serialize-and-deserialize-binary-tree/
# Companies: Uber,Google,Apple,Quora,LinkedIn,Amazon,Expedia,Facebook,Oracle,Microsoft
# Categories: Tree,Design

# ----------------------------------------------------------------------------

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Codec:

    def serialize(self, root):
        def ser(node, res=[]):
            if not node:
                res.append('#')
            else:
                res.append(node.val)
                ser(node.left)
                ser(node.right)
            return res
        return str(ser(root))



    def deserialize(self, data):
        data = eval(data)
        self.i = 0
        def deser():
            node = None
            self.i += 1
            if data[self.i - 1] != "#":
                node = TreeNode(data[self.i - 1])
                l,r = deser(), deser()
                node.left, node.right = l, r
            return node

        return deser()




# Your Codec object will be instantiated and called as such:
# codec = Codec()
# codec.deserialize(codec.serialize(root))


# ============================================================================

# 298. Binary Tree Longest Consecutive Sequence
# Difficulty: Medium
# link: https://leetcode.com/problems/binary-tree-longest-consecutive-sequence/
# Companies: Amazon,Google
# Categories: Tree

# ----------------------------------------------------------------------------

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def longestConsecutive(self, root):

        def _cnt(node, par, cnt):
            if not node: return 0
            prev_cnt = cnt
            cnt = cnt + 1 if (node.val == par.val + 1) else 1
            l, r = _cnt(node.left, node, cnt), _cnt(node.right, node, cnt)
            return max(l, r, cnt, prev_cnt)
        return _cnt(root, root, 1)


# ============================================================================

# 299. Bulls and Cows
# Difficulty: Easy
# link: https://leetcode.com/problems/bulls-and-cows/
# Companies: Amazon,Google
# Categories: Hash Table

# ----------------------------------------------------------------------------

class Solution(object):
    def getHint(self, secret, guess):
        """
        :type secret: str
        :type guess: str
        :rtype: str
        """
        a = 0
        s_counts = {}
        g_counts = {}
        for s_c, g_c in zip(secret, guess):
            if s_c == g_c: a += 1
            else:
                s_counts[s_c] = s_counts.get(s_c, 0) + 1
                g_counts[g_c] = g_counts.get(g_c, 0) + 1
        b = sum(min(s_counts.get(g_c,0), g_counts[g_c]) for g_c in g_counts)
        return "%dA%dB" %(a, b)


# ============================================================================

# 300. Longest Increasing Subsequence
# Difficulty: Medium
# link: https://leetcode.com/problems/longest-increasing-subsequence/
# Companies: Amazon,Google,Microsoft
# Categories: Binary Search,Dynamic Programming

# ----------------------------------------------------------------------------

class Solution(object):
    def lengthOfLIS(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        end_idx = [None] * len(nums)
        length = 0
        for i, num in enumerate(nums):
            j = 0
            while j < length and nums[end_idx[j]] < num:
                j += 1
            end_idx[j] = i
            length = max(j + 1, length)
        return length


# ============================================================================

# 303. Range Sum Query - Immutable
# Difficulty: Easy
# link: https://leetcode.com/problems/range-sum-query-immutable/
# Companies: Facebook,Bloomberg
# Categories: Dynamic Programming

# ----------------------------------------------------------------------------

class NumArray(object):

    def __init__(self, nums):
        self.l_sum = l_sum = nums
        for i in range(1, len(l_sum)): l_sum[i] = l_sum[i - 1] + l_sum[i]


    def sumRange(self, i, j):
        return self.l_sum[j] - (self.l_sum[i - 1] if i > 0 else 0)


# Your NumArray object will be instantiated and called as such:
# obj = NumArray(nums)
# param_1 = obj.sumRange(i,j)


# ============================================================================

# 304. Range Sum Query 2D - Immutable
# Difficulty: Medium
# link: https://leetcode.com/problems/range-sum-query-2d-immutable/
# Companies: Houzz,Amazon,Google,Facebook,VMware
# Categories: Dynamic Programming

# ----------------------------------------------------------------------------

class NumMatrix(object):

    def __init__(self, matrix):
        """
        :type matrix: List[List[int]]
        """
        self.matrix = matrix
        for i in range(len(matrix)):
            for j in range(len(matrix[0]) if matrix else 0):
                matrix[i][j] = self.sub_points([[i - 1, j], [i, j - 1], [i, j]], [[i - 1, j - 1]])

    def sub_points(self, coords, sub_coords):
        def sum_points(coords):
            return sum(self.matrix[x][y] for x, y in coords if x >=0 and y >= 0)
        return sum_points(coords) - sum_points(sub_coords)


    def sumRegion(self, row1, col1, row2, col2):
        """
        :type row1: int
        :type col1: int
        :type row2: int
        :type col2: int
        :rtype: int
        """
        return self.sub_points([[row2, col2], [row1 - 1, col1 - 1]],[[row1 - 1, col2], [row2, col1 - 1]])



# Your NumMatrix object will be instantiated and called as such:
# obj = NumMatrix(matrix)
# param_1 = obj.sumRegion(row1,col1,row2,col2)


# ============================================================================

# 306. Additive Number
# Difficulty: Medium
# link: https://leetcode.com/problems/additive-number/
# Companies: Epic Systems
# Categories: Backtracking

# ----------------------------------------------------------------------------

class Solution(object):
    def isAdditiveNumber(self, num):
        def is_seq(i, j, k):
            if k == len(num): return True
            a, b = int(num[i:j]), int(num[j:k])
            if len(str(a)) != j - i or len(str(b)) != k - j: return False
            total = str(a + b)
            return num[k:].startswith(total) and is_seq(j, k, k + len(total))
        return any(is_seq(0, i, j)
                   for j in range(2, len(num))
                   for i in range(1, j))


# ============================================================================

# 307. Range Sum Query - Mutable
# Difficulty: Medium
# link: https://leetcode.com/problems/range-sum-query-mutable/
# Companies: Twitter,Google
# Categories: Binary Indexed Tree,Segment Tree

# ----------------------------------------------------------------------------

class NumArray(object):

    def __init__(self, nums):
        self.update = nums.__setitem__
        self.sumRange = lambda i, j: sum(nums[i:j+1])


# Your NumArray object will be instantiated and called as such:
# obj = NumArray(nums)
# obj.update(i,val)
# param_2 = obj.sumRange(i,j)


# ============================================================================

# 310. Minimum Height Trees
# Difficulty: Medium
# link: https://leetcode.com/problems/minimum-height-trees/
# Companies: Google
# Categories: Breadth-first Search,Graph

# ----------------------------------------------------------------------------

class Solution(object):

    # BFS from leave nodes
    def findMinHeightTrees(self, n, edges):
        if n == 1: return [0]
        graph = [set() for _ in range(n)]
        for a, b in edges:
            graph[a].add(b)
            graph[b].add(a)
        # leaf nodes
        bfs = [node for node, adjs in enumerate(graph) if len(adjs) == 1]
        while n > 2:
            n -= len(bfs)
            new_bfs = []
            for node in bfs:
                parent = graph[node].pop()
                graph[parent].remove(node)
                if len(graph[parent]) == 1: new_bfs.append(parent)
            bfs = new_bfs
        return bfs


    # Cutting leaf nodes until there's 2 or less
    def __findMinHeightTrees(self, n, edges):
        graph = {i:set() for i in xrange(n)}
        for a, b in edges:
            graph[a].add(b)
            graph[b].add(a)
        # iteratively remove leaves until 1/2 nodes left
        visited = set()
        while n - len(visited) > 2:
            leaves = [node for node, adjs in graph.iteritems() if len(adjs) == 1]
            for leave in leaves:
                for adj in graph[leave]:
                    if leave in graph[adj]:
                        graph[adj].remove(leave)
                del graph[leave]
            visited.update(set(leaves))
        return list(set(range(n)) - visited)


# ============================================================================

# 314. Binary Tree Vertical Order Traversal
# Difficulty: Medium
# link: https://leetcode.com/problems/binary-tree-vertical-order-traversal/
# Companies: Databricks,Amazon,Facebook,ByteDance
# Categories: Hash Table

# ----------------------------------------------------------------------------

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def verticalOrder(self, root):
        if not root: return []
        res = {}
        bfs = [(root, 0)]
        while bfs:
            for node, offset in bfs:
                res.setdefault(offset, []).append(node.val)
            bfs = [(child, child_offset) for node, offset in bfs for child, child_offset in [[node.left, offset - 1], [node.right, offset + 1]] if child]
        min_offset, max_offset = abs(min(res.keys())), max(res.keys())
        return [res[i - min_offset] for i in range(max_offset + min_offset + 1)]


# ============================================================================

# 318. Maximum Product of Word Lengths
# Difficulty: Medium
# link: https://leetcode.com/problems/maximum-product-of-word-lengths/
# Companies: Google
# Categories: Bit Manipulation

# ----------------------------------------------------------------------------

class Solution(object):
    def maxProduct(self, words):
        """
        :type words: List[str]
        :rtype: int
        """
        bit_wise_words = []
        for word in words:
            int_word = 0
            for char in word:
                int_word |= 1 << ord(char) % 26
            bit_wise_words.append(int_word)

        max_length = 0
        for i in range(len(words)):
            for j in range(i, len(words)):
                length = len(words[i]) * len(words[j])
                if bit_wise_words[i] & bit_wise_words[j] == 0 and max_length < length:
                    max_length = length
        return max_length


# ============================================================================

# 319. Bulb Switcher
# Difficulty: Medium
# link: https://leetcode.com/problems/bulb-switcher/
# Companies: Facebook
# Categories: Math,Brainteaser

# ----------------------------------------------------------------------------

class Solution(object):
    def bulbSwitch(self, n):
        """
        :type n: int
        :rtype: int
        """
        return int(n ** (1.0/2))


# ============================================================================

# 322. Coin Change
# Difficulty: Medium
# link: https://leetcode.com/problems/coin-change/
# Companies: Adobe,Apple,Affirm,Yahoo,Airbnb,Amazon,Bloomberg,Capital One,Microsoft
# Categories: Dynamic Programming

# ----------------------------------------------------------------------------

class Solution(object):
    def coinChange(self, coins, amount):
        """
        :type coins: List[int]
        :type amount: int
        :rtype: int
        """
        dp = [float('inf')] * (amount + 1)
        dp[0] = 0
        for i in range(1, len(dp)):
            for coin in coins:
                prev_idx = i - coin
                if prev_idx >= 0:
                    dp[i] = min(dp[prev_idx] + 1, dp[i])
        return dp[amount] if type(dp[amount]) == int else -1


# ============================================================================

# 326. Power of Three
# Difficulty: Easy
# link: https://leetcode.com/problems/power-of-three/
# Companies: Hulu,Google,Goldman Sachs
# Categories: Math

# ----------------------------------------------------------------------------

class Solution(object):
    def isPowerOfThree(self, n):
        # power = 1
        # while 3 ** (power + 1) <= (((1 << 32) - 1)):
        #     power += 1
        # print 3 ** power

        return n > 0 and 4052555153018976267 % n == 0


# ============================================================================

# 328. Odd Even Linked List
# Difficulty: Medium
# link: https://leetcode.com/problems/odd-even-linked-list/
# Companies: Google,Microsoft,Bloomberg,Capital One
# Categories: Linked List

# ----------------------------------------------------------------------------

# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def oddEvenList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        even_odd_head = [ListNode("even"), ListNode("odd")]
        even_odd = even_odd_head[:]
        cur = head
        toggle = 0
        while cur:
            even_odd[toggle].next = ListNode(cur.val)
            even_odd[toggle] = even_odd[toggle].next
            toggle = 1 - toggle
            cur = cur.next
        even_odd[0].next = even_odd_head[1].next
        return even_odd_head[0].next


# ============================================================================

# 329. Longest Increasing Path in a Matrix
# Difficulty: Hard
# link: https://leetcode.com/problems/longest-increasing-path-in-a-matrix/
# Companies: Adobe,Amazon,Google,Facebook,Apple
# Categories: Depth-first Search,Topological Sort,Memoization

# ----------------------------------------------------------------------------

class Solution(object):
    def longestIncreasingPath(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: int
        """
        xy_dir = zip([-1, 0, 1, 0], [0, -1, 0, 1])
        m, n = len(matrix), len(matrix[0]) if matrix else 0
        dp = [[None] * n for _ in range(m)]
        def get_max_inc(x, y):
            if dp[x][y] is not None: return dp[x][y]
            adj = [(i, j) for i, j in [(x + x_d, y + y_d) for x_d, y_d in xy_dir]
                   if 0 <= i < m and 0 <= j < n and matrix[x][y] > matrix[i][j]]
            dp[x][y] = max([(get_max_inc(i, j) + 1) for i, j in adj] or [1])
            return dp[x][y]
        return max([get_max_inc(x, y) for x in range(m) for y in range(n)] or [0])


# ============================================================================

# 331. Verify Preorder Serialization of a Binary Tree
# Difficulty: Medium
# link: https://leetcode.com/problems/verify-preorder-serialization-of-a-binary-tree/
# Companies: Google
# Categories: Stack

# ----------------------------------------------------------------------------

class Solution(object):
    def isValidSerialization(self, preorder):
        preorder = preorder.split(',')
        diff = 1
        for node in preorder:
            diff -= 1
            if diff < 0: return False
            if node != '#': diff += 2
        return diff == 0


# ============================================================================

# 332. Reconstruct Itinerary
# Difficulty: Medium
# link: https://leetcode.com/problems/reconstruct-itinerary/
# Companies: Uber,Google,Apple,Yelp,Goldman Sachs,Amazon,Bloomberg,Microsoft,Twilio
# Categories: Depth-first Search,Graph

# ----------------------------------------------------------------------------

class Solution(object):
    def findItinerary(self, tickets):
        G = {}
        for f, t in tickets:
            G.setdefault(f, []).append(t)

        for node in G: G[node].sort(reverse=True)

        def visit(airport, route=[]):
            while G.get(airport, []):
                visit(G[airport].pop(), route)
            route.append(airport)
            return route
        return reversed(visit('JFK'))


# ============================================================================

# 334. Increasing Triplet Subsequence
# Difficulty: Medium
# link: https://leetcode.com/problems/increasing-triplet-subsequence/
# Companies: Google
# Categories:

# ----------------------------------------------------------------------------

class Solution(object):
    def increasingTriplet(self, nums):
        first = second = float('inf')
        for n in nums:
            if n <= first:
                first = n
            elif n <= second:
                second = n
            else:
                return True
        return False


# ============================================================================

# 337. House Robber III
# Difficulty: Medium
# link: https://leetcode.com/problems/house-robber-iii/
# Companies: Amazon,Google
# Categories: Tree,Depth-first Search

# ----------------------------------------------------------------------------

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def rob(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        # Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def rob(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        def _rob(node):
            if not node: return 0, 0
            inc_l, not_inc_l = _rob(node.left)
            inc_r, not_inc_r = _rob(node.right)
            inc_node = node.val + not_inc_l + not_inc_r
            not_inc_node = max(inc_l, not_inc_l) + max(inc_r, not_inc_r)
            return inc_node, not_inc_node
        return max(_rob(root))


# ============================================================================

# 338. Counting Bits
# Difficulty: Medium
# link: https://leetcode.com/problems/counting-bits/
# Companies: Mathworks,Amazon,Apple
# Categories: Dynamic Programming,Bit Manipulation

# ----------------------------------------------------------------------------

class Solution(object):
    def countBits(self, num):
        """
        :type num: int
        :rtype: List[int]
        """
        res = [0]
        for i in range(1, num + 1):
            res.append(res[i >> 1] + (i & 1))
        return res


# ============================================================================

# 340. Longest Substring with At Most K Distinct Characters
# Difficulty: Hard
# link: https://leetcode.com/problems/longest-substring-with-at-most-k-distinct-characters/
# Companies: Uber,Google,Amazon,Facebook,Bloomberg,Microsoft
# Categories: Hash Table,String,Sliding Window

# ----------------------------------------------------------------------------

class Solution(object):
    def lengthOfLongestSubstringKDistinct(self, s, k):
        from collections import Counter
        cnt = Counter()
        res = i = 0
        for j, char in enumerate(s):
            cnt[char] += 1
            while len(cnt) > k:
                cnt[s[i]] -= 1
                if not cnt[s[i]]:
                    del cnt[s[i]]
                i+= 1
            res = max(res, j - i + 1)
        return res


# ============================================================================

# 341. Flatten Nested List Iterator
# Difficulty: Medium
# link: https://leetcode.com/problems/flatten-nested-list-iterator/
# Companies: Uber,Lyft,Apple,Twitter,LinkedIn,Airbnb,Amazon,Facebook
# Categories: Stack,Design

# ----------------------------------------------------------------------------

# """
# This is the interface that allows for creating nested lists.
# You should not implement it, or speculate about its implementation
# """
#class NestedInteger(object):
#    def isInteger(self):
#        """
#        @return True if this NestedInteger holds a single integer, rather than a nested list.
#        :rtype bool
#        """
#
#    def getInteger(self):
#        """
#        @return the single integer that this NestedInteger holds, if it holds a single integer
#        Return None if this NestedInteger holds a nested list
#        :rtype int
#        """
#
#    def getList(self):
#        """
#        @return the nested list that this NestedInteger holds, if it holds a nested list
#        Return None if this NestedInteger holds a single integer
#        :rtype List[NestedInteger]
#        """

class NestedIterator(object):

    def __init__(self, nestedList):
        self.stack, self.cache = [[nestedList, 0]], None

    def next(self):
        self.hasNext()
        res, self.cache = self.cache, None
        return res


    def hasNext(self):
        """
        :rtype: bool
        """
        if self.cache is not None: return True
        elif not self.stack: return False
        next_lst, next_idx = self.stack[-1]
        if next_idx < len(next_lst):
            if next_lst[next_idx].isInteger():
                self.cache = next_lst[next_idx].getInteger()
                self.stack[-1][1] += 1
                return True
            else:
                self.stack[-1][1] += 1
                self.stack.append([next_lst[next_idx].getList(), 0])
                return self.hasNext()
        else:
            self.stack.pop()
            return self.hasNext()


# Your NestedIterator object will be instantiated and called as such:
# i, v = NestedIterator(nestedList), []
# while i.hasNext(): v.append(i.next())


# ============================================================================

# 342. Power of Four
# Difficulty: Easy
# link: https://leetcode.com/problems/power-of-four/
# Companies: Uber,Two Sigma
# Categories: Bit Manipulation

# ----------------------------------------------------------------------------

class Solution(object):
    def isPowerOfFour(self, num):
        """
        :type num: int
        :rtype: bool
        """
        # mask = 0
        # while (mask << 2 | 1) < ((1 << 32) - 1):
        #     mask = mask << 2 | 1
        # print mask
        return (num & (num - 1)) == 0 and bool(1431655765 & num)


# ============================================================================

# 343. Integer Break
# Difficulty: Medium
# link: https://leetcode.com/problems/integer-break/
# Companies: Apple
# Categories: Math,Dynamic Programming

# ----------------------------------------------------------------------------

class Solution(object):
    # bottom up
    def integerBreak(self, n):
        dp = [0] * (n + 1)
        dp[1] = 1
        for i in range(2, n + 1):
            dp[i] = max((i - j) * max(dp[j], j) for j in range(1, i))
        return dp[n]


# ============================================================================

# 344. Reverse String
# Difficulty: Easy
# link: https://leetcode.com/problems/reverse-string/
# Companies: Google,Adobe,Apple,eBay,Amazon,Facebook,Bloomberg,Microsoft
# Categories: Two Pointers,String

# ----------------------------------------------------------------------------

class Solution(object):
    def reverseString(self, s):
        """
        :type s: str
        :rtype: str
        """
        return s[::-1]


# ============================================================================

# 345. Reverse Vowels of a String
# Difficulty: Easy
# link: https://leetcode.com/problems/reverse-vowels-of-a-string/
# Companies: Amazon
# Categories: Two Pointers,String

# ----------------------------------------------------------------------------

class Solution(object):
    def reverseVowels(self, s):
        """
        :type s: str
        :rtype: str
        """
        vowels = [char for char in s if char in 'aeiouAEIOU']
        return ''.join([ (char if char not in 'aeiouAEIOU' else vowels.pop()) for char in s])


# ============================================================================

# 347. Top K Frequent Elements
# Difficulty: Medium
# link: https://leetcode.com/problems/top-k-frequent-elements/
# Companies: Uber,Google,Apple,Yelp,Snapchat,Amazon,Facebook,Oracle,Microsoft
# Categories: Hash Table,Heap

# ----------------------------------------------------------------------------

class Solution(object):
    def topKFrequent(self, nums, k):
        from collections import Counter
        import heapq
        freq_to_nums = [(-cnt, num) for num, cnt in  Counter(nums).iteritems()]
        heapq.heapify(freq_to_nums)
        return [heapq.heappop(freq_to_nums)[1] for _ in range(k)]


# ============================================================================

# 349. Intersection of Two Arrays
# Difficulty: Easy
# link: https://leetcode.com/problems/intersection-of-two-arrays/
# Companies: Oracle,Lyft,Facebook,Microsoft,LinkedIn
# Categories: Hash Table,Two Pointers,Binary Search,Sort

# ----------------------------------------------------------------------------

class Solution(object):
    def intersection(self, nums1, nums2): return list(set(nums1) & set(nums2))


# ============================================================================

# 350. Intersection of Two Arrays II
# Difficulty: Easy
# link: https://leetcode.com/problems/intersection-of-two-arrays-ii/
# Companies: Amazon,Google,Facebook,Apple,LinkedIn
# Categories: Hash Table,Two Pointers,Binary Search,Sort

# ----------------------------------------------------------------------------

class Solution(object):
    def intersect(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: List[int]
        """
        from collections import Counter
        counts1 = Counter(nums1)
        res = []
        for num in nums2:
            if counts1.get(num, 0):
                res.append(num)
                counts1[num] -= 1
        return res


# ============================================================================

# 352. Data Stream as Disjoint Intervals
# Difficulty: Hard
# link: https://leetcode.com/problems/data-stream-as-disjoint-intervals/
# Companies: Amazon
# Categories: Binary Search,Ordered Map

# ----------------------------------------------------------------------------

# Definition for an interval.
# class Interval(object):
#     def __init__(self, s=0, e=0):
#         self.start = s
#         self.end = e

class SummaryRanges(object):

    def __init__(self):
        self.start_times = set()
        self.start_time_to_num_of_elems = {}

    def addNum(self, num):
        consecutive = self.start_time_to_num_of_elems
        start_times = self.start_times
        if num not in consecutive:
            size = 1
            left = right = None
            if num - 1 in consecutive:
                size += consecutive[num - 1]
                left = (num - 1) - (consecutive[num - 1] - 1)
            if num + 1 in consecutive:
                start_times.remove(num + 1)
                size += consecutive[num + 1]
                right = (num + 1) + (consecutive[num + 1] - 1)
                consecutive[right] = size
            if left is not None:
                consecutive[left] = size
                start_times.add(left)
            else:
                start_times.add(num)

            consecutive[num] = size

    def getIntervals(self):
        res = []
        for start_time in sorted(self.start_times):
            res.append([start_time, start_time + self.start_time_to_num_of_elems[start_time] - 1])
        return res if res else None


# Your SummaryRanges object will be instantiated and called as such:
# obj = SummaryRanges()
# obj.addNum(val)
# param_2 = obj.getIntervals()


# ============================================================================

# 354. Russian Doll Envelopes
# Difficulty: Hard
# link: https://leetcode.com/problems/russian-doll-envelopes/
# Companies: Google,Microsoft
# Categories: Binary Search,Dynamic Programming

# ----------------------------------------------------------------------------

class Solution {
public int maxEnvelopes(int[][] envelopes) {
        // sort the widths and then longest increasing subseq problem for heights
        if (envelopes == null || envelopes.length == 0) {
            return 0;
        }
        Comparator comp = Comparator.comparing((int[] arr)-> arr[0]).thenComparing((int[] arr)->arr[1], Comparator.reverseOrder());
        Arrays.sort(envelopes, comp);
        int result = 1;
        int[] dp = new int[envelopes.length];
        Arrays.fill(dp, 1);
        for (int i = 1; i < envelopes.length; i ++) {
            for (int j = 0; j < i ; j ++) {
                if (envelopes[i][0] > envelopes[j][0] && envelopes[i][1] > envelopes[j][1]) {
                    dp[i] = Math.max(dp[i], 1 + dp[j]);
                }
            }
            result = Math.max(dp[i], result);
        }
        return result;
    }
}


# ============================================================================

# 356. Line Reflection
# Difficulty: Medium
# link: https://leetcode.com/problems/line-reflection/
# Companies: Amazon,Google
# Categories: Hash Table,Math

# ----------------------------------------------------------------------------

class Solution(object):
    def isReflected(self, points):
        if not points: return True
        l, r = min(x for x, y in points), max(x for x, y in points)
        seen = set(map(tuple, points))
        for p in points:
            if (l + r - p[0], p[1]) not in seen: return False
        return True


# ============================================================================

# 366. Find Leaves of Binary Tree
# Difficulty: Medium
# link: https://leetcode.com/problems/find-leaves-of-binary-tree/
# Companies: LinkedIn
# Categories: Tree,Depth-first Search

# ----------------------------------------------------------------------------

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def findLeaves(self, root):
        def dfs(node):
            if not node: return -1
            l, r = dfs(node.left), dfs(node.right)
            idx = max(l, r) + 1
            if len(self.res) <= idx:
                self.res.append([])
            self.res[idx].append(node.val)
            return idx
        self.res = []
        dfs(root)
        return self.res


# ============================================================================

# 367. Valid Perfect Square
# Difficulty: Easy
# link: https://leetcode.com/problems/valid-perfect-square/
# Companies: Microsoft,LinkedIn
# Categories: Math,Binary Search

# ----------------------------------------------------------------------------

class Solution(object):
    def isPerfectSquare(self, num):
        """
        :type num: int
        :rtype: bool
        """
        diff = 3
        sq = 1
        while sq < num:
            sq += diff
            diff += 2
        return sq == num


# ============================================================================

# 373. Find K Pairs with Smallest Sums
# Difficulty: Medium
# link: https://leetcode.com/problems/find-k-pairs-with-smallest-sums/
# Companies: LinkedIn
# Categories: Heap

# ----------------------------------------------------------------------------

class Solution(object):
    def kSmallestPairs(self, nums1, nums2, k):
        import heapq
        if not nums1 or not nums2: return []
        heap = [(nums1[0] + nums2[0], 0, 0)]
        res = []
        visited = set()
        for _ in range(k):
            if not heap: break
            _, i, j = heapq.heappop(heap)
            if i + 1 < len(nums1) and ((i + 1, j) not in visited):
                heapq.heappush(heap, (nums1[i + 1] + nums2[j], i + 1, j))
                visited.add((i + 1, j))
            if j + 1 < len(nums2) and ((i, j + 1) not in visited):
                heapq.heappush(heap, (nums1[i] + nums2[j + 1], i, j + 1))
                visited.add((i, j + 1))
            res.append([nums1[i], nums2[j]])
        return res


# ============================================================================

# 374. Guess Number Higher or Lower
# Difficulty: Easy
# link: https://leetcode.com/problems/guess-number-higher-or-lower/
# Companies: Google
# Categories: Binary Search

# ----------------------------------------------------------------------------

# The guess API is already defined for you.
# @param num, your guess
# @return -1 if my number is lower, 1 if my number is higher, otherwise return 0
# def guess(num):

class Solution(object):
    def guessNumber(self, n):
        low, high = 1, n
        while low <= high:
            mid = (low + high) / 2
            if guess(mid) == 0: return mid
            elif guess(mid) > 0: low = mid + 1
            else: high = mid - 1


# ============================================================================

# 376. Wiggle Subsequence
# Difficulty: Medium
# link: https://leetcode.com/problems/wiggle-subsequence/
# Companies: Amazon
# Categories: Dynamic Programming,Greedy

# ----------------------------------------------------------------------------

class Solution(object):
    def wiggleMaxLength(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        # count the up_down seq,
        # if the first seq is down_up,
        # then we add one to the result
        if not nums: return 0
        count = 1
        dir_tog = None
        for i in range(1, len(nums)):
            if dir_tog is None and nums[i-1] != nums[i]:
                dir_tog = nums[i-1] < nums[i]
            if (dir_tog is not None) and \
             (nums[i-1] < nums[i] and dir_tog or \
              nums[i-1] > nums[i] and not dir_tog):
                count, dir_tog = count + 1, not dir_tog
        return count


# ============================================================================

# 377. Combination Sum IV
# Difficulty: Medium
# link: https://leetcode.com/problems/combination-sum-iv/
# Companies: Amazon
# Categories: Dynamic Programming

# ----------------------------------------------------------------------------

class Solution(object):
    def combinationSum4(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        dp = {0: 1}
        def _combinationSum4(target):
            if target in dp: return dp[target]
            res = 0
            for num in nums:
                if target - num >= 0:
                    res += _combinationSum4(target - num)
            dp[target] = res
            return res
        return _combinationSum4(target)


# ============================================================================

# 378. Kth Smallest Element in a Sorted Matrix
# Difficulty: Medium
# link: https://leetcode.com/problems/kth-smallest-element-in-a-sorted-matrix/
# Companies: Walmart Labs,Amazon,Facebook,Microsoft,Apple
# Categories: Binary Search,Heap

# ----------------------------------------------------------------------------

class Solution(object):
    def kthSmallest(self, matrix, k):
        """
        :type matrix: List[List[int]]
        :type k: int
        :rtype: int
        """
        return sorted(i for row in matrix for i in row)[k-1]


# ============================================================================

# 380. Insert Delete GetRandom O(1)
# Difficulty: Medium
# link: https://leetcode.com/problems/insert-delete-getrandom-o1/
# Companies: Uber,Google,Apple,Affirm,Quora,Databricks,LinkedIn,Goldman Sachs,Amazon,Facebook,Yandex,Bloomberg,Microsoft
# Categories: Array,Hash Table,Design

# ----------------------------------------------------------------------------

import random

class RandomizedSet(object):

    def __init__(self):
        self.nums, self.pos = [], {}

    def insert(self, val):
        if val not in self.pos:
            self.nums.append(val)
            self.pos[val] = len(self.nums) - 1
            return True
        return False


    def remove(self, val):
        if val in self.pos:
            idx, last = self.pos[val], self.nums[-1]
            self.nums[idx], self.pos[last] = last, idx
            self.nums.pop()
            del self.pos[val]
            return True
        return False

    def getRandom(self):
        return self.nums[random.randint(0, len(self.nums) - 1)]


# ============================================================================

# 382. Linked List Random Node
# Difficulty: Medium
# link: https://leetcode.com/problems/linked-list-random-node/
# Companies: Google
# Categories: Reservoir Sampling

# ----------------------------------------------------------------------------

# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None
import random
class Solution(object):

    def __init__(self, head):
        """
        @param head The linked list's head.
        Note that the head is guaranteed to be not null, so it contains at least one node.
        :type head: ListNode
        """
        self.num_elem = 0
        self.head = cur = head
        while cur:
            self.num_elem += 1
            cur = cur.next

    def getRandom(self):
        """
        Returns a random node's value.
        :rtype: int
        """
        idx = random.randint(0, self.num_elem - 1)
        cur = self.head
        while cur:
            if idx == 0:
                return cur.val
            idx -= 1
            cur = cur.next




# Your Solution object will be instantiated and called as such:
# obj = Solution(head)
# param_1 = obj.getRandom()


# ============================================================================

# 383. Ransom Note
# Difficulty: Easy
# link: https://leetcode.com/problems/ransom-note/
# Companies: Microsoft
# Categories: String

# ----------------------------------------------------------------------------

class Solution(object):
    def canConstruct(self, ransomNote, magazine):
        """
        :type ransomNote: str
        :type magazine: str
        :rtype: bool
        """
        counts = [0] * 26
        for char in magazine: counts[ord(char) % len(counts)] += 1
        for char in ransomNote: counts[ord(char) % len(counts)] -= 1
        return all(count >= 0 for count in counts)


# ============================================================================

# 384. Shuffle an Array
# Difficulty: Medium
# link: https://leetcode.com/problems/shuffle-an-array/
# Companies: Yahoo,Amazon,Microsoft,Apple
# Categories:

# ----------------------------------------------------------------------------

from random import randint
class Solution(object):
    def __init__(self, nums):
        self.nums = nums
    def reset(self):
        return self.nums
    def shuffle(self):
        self.rand_nums = self.nums[:]
        for i in range(len(self.rand_nums)):
            swap_idx = randint(0, len(self.rand_nums) - 1)
            self.rand_nums[i], self.rand_nums[swap_idx] = self.rand_nums[swap_idx], self.rand_nums[i]
        return self.rand_nums





# Your Solution object will be instantiated and called as such:
# obj = Solution(nums)
# param_1 = obj.reset()
# param_2 = obj.shuffle()


# ============================================================================

# 385. Mini Parser
# Difficulty: Medium
# link: https://leetcode.com/problems/mini-parser/
# Companies: Google
# Categories: String,Stack

# ----------------------------------------------------------------------------

# """
# This is the interface that allows for creating nested lists.
# You should not implement it, or speculate about its implementation
# """
#class NestedInteger(object):
#    def __init__(self, value=None):
#        """
#        If value is not specified, initializes an empty list.
#        Otherwise initializes a single integer equal to value.
#        """
#
#    def isInteger(self):
#        """
#        @return True if this NestedInteger holds a single integer, rather than a nested list.
#        :rtype bool
#        """
#
#    def add(self, elem):
#        """
#        Set this NestedInteger to hold a nested list and adds a nested integer elem to it.
#        :rtype void
#        """
#
#    def setInteger(self, value):
#        """
#        Set this NestedInteger to hold a single integer equal to value.
#        :rtype void
#        """
#
#    def getInteger(self):
#        """
#        @return the single integer that this NestedInteger holds, if it holds a single integer
#        Return None if this NestedInteger holds a nested list
#        :rtype int
#        """
#
#    def getList(self):
#        """
#        @return the nested list that this NestedInteger holds, if it holds a nested list
#        Return None if this NestedInteger holds a single integer
#        :rtype List[NestedInteger]
#        """

class Solution(object):
    def deserialize(self, s):
        """
        :type s: str
        :rtype: NestedInteger
        """
        i = 0
        stack = [NestedInteger()]
        while i < len(s):
            char = s[i]
            if char == '[':
                ni = NestedInteger()
                stack[-1].add(ni)
                stack.append(ni)
            elif char.isdigit() or char == '-':
                j = i
                while j < len(s) and (s[j] not in ',]'): j += 1
                ni = NestedInteger(int(s[i:j]))
                stack[-1].add(ni)
                i = j - 1
            elif char == ']':
                stack.pop()
            i += 1

        return  stack[0].getList()[0]


# ============================================================================

# 387. First Unique Character in a String
# Difficulty: Easy
# link: https://leetcode.com/problems/first-unique-character-in-a-string/
# Companies: Google,Apple,Goldman Sachs,Amazon,Facebook,Bloomberg,Microsoft,Zillow
# Categories: Hash Table,String

# ----------------------------------------------------------------------------

class Solution(object):
    def firstUniqChar(self, s):
        """
        :type s: str
        :rtype: int
        """
        from collections import Counter
        counts = Counter(s)
        for i, char in enumerate(s):
            if counts[char] == 1:
                return i
        return -1


# ============================================================================

# 389. Find the Difference
# Difficulty: Easy
# link: https://leetcode.com/problems/find-the-difference/
# Companies: Google
# Categories: Hash Table,Bit Manipulation

# ----------------------------------------------------------------------------

class Solution(object):
    def findTheDifference(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: str
        """
        return chr(reduce(operator.xor, [ord(char) for char in s + t], 0))


# ============================================================================

# 391. Perfect Rectangle
# Difficulty: Hard
# link: https://leetcode.com/problems/perfect-rectangle/
# Companies: Apple
# Categories: Line Sweep

# ----------------------------------------------------------------------------

class Solution(object):
    def isRectangleCover(self, rectangles):
        """
        :type rectangles: List[List[int]]
        :rtype: bool
        """
        x1 = y1 = float('inf')
        x2 = y2 = float('-inf')
        area = 0
        pnt_cnt = {}

        for rect in rectangles:
            a1, b1, a2, b2 = rect
            x1, y1 = min(x1, a1), min(y1, b1)
            x2, y2 = max(x2, a2), max(y2, b2)
            for x, y in [(a1, b1), (a2, b2), (a1, b2), (a2, b1)]: pnt_cnt[(x, y)] = pnt_cnt.get((x, y), 0) + 1
            area += (b2 - b1) * (a2 - a1)

        if area != (y2 - y1) * (x2 - x1): return False

        for pnt, cnt in pnt_cnt.items():
            if pnt in [(x1, y1), (x2, y2), (x1, y2), (x2, y1)]:
                if pnt_cnt.get(pnt, 0) != 1: return False
            elif cnt % 2 != 0: return False
        return True


# ============================================================================

# 392. Is Subsequence
# Difficulty: Easy
# link: https://leetcode.com/problems/is-subsequence/
# Companies: Google,Pinterest
# Categories: Binary Search,Dynamic Programming,Greedy

# ----------------------------------------------------------------------------

class Solution(object):
    def isSubsequence(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: bool
        """
        i = 0
        for char in s:
            while i < len(t) and t[i] != char:
                i += 1
            i += 1
            if i > len(t):
                return False
        return True


# ============================================================================

# 394. Decode String
# Difficulty: Medium
# link: https://leetcode.com/problems/decode-string/
# Companies: Google,Apple,Atlassian,Amazon,Salesforce,Facebook,Bloomberg,Oracle,Tencent,VMware
# Categories: Stack,Depth-first Search

# ----------------------------------------------------------------------------

class Solution(object):
    def decodeString(self, s):
        """
        :type s: str
        :rtype: str
        """
        stack = []
        str_queue = []
        def add_to_queue(str_queue, stack):
            stack.append(''.join(str_queue))
            del str_queue[:]
            last = []
            while stack and not stack[-1].isdigit(): last.append(stack.pop())
            if last: stack.append(''.join(list(reversed(last))))

        for char in s:
            if '[' == char:
                add_to_queue(str_queue, stack)
            elif ']' == char:
                add_to_queue(str_queue, stack)
                substr, num = stack.pop(), int(stack.pop())
                stack.append(substr * num)
            elif str_queue and str_queue[-1].isdigit() != char.isdigit():
                add_to_queue(str_queue, stack)
                str_queue.append(char)
            else: str_queue.append(char)
        stack.append(''.join(str_queue))
        return ''.join(stack)


# ============================================================================

# 395. Longest Substring with At Least K Repeating Characters
# Difficulty: Medium
# link: https://leetcode.com/problems/longest-substring-with-at-least-k-repeating-characters/
# Companies: Uber,Amazon,Google
# Categories:

# ----------------------------------------------------------------------------

class Solution(object):
    def longestSubstring(self, s, k):
        """
        :type s: str
        :type k: int
        :rtype: int
        """
        for char in set(s):
            if s.count(char) < k: return max(self.longestSubstring(t, k) for t in s.split(char))
        return len(s)


# ============================================================================

# 398. Random Pick Index
# Difficulty: Medium
# link: https://leetcode.com/problems/random-pick-index/
# Companies: Facebook
# Categories: Reservoir Sampling

# ----------------------------------------------------------------------------

class Solution(object):

    def __init__(self, nums):
        self.num_to_idx = {}
        for i, num in enumerate(nums): self.num_to_idx.setdefault(num, []).append(i)

    def pick(self, target):
        import random
        return random.choice(self.num_to_idx[target])


# Your Solution object will be instantiated and called as such:
# obj = Solution(nums)
# param_1 = obj.pick(target)


# ============================================================================

# 399. Evaluate Division
# Difficulty: Medium
# link: https://leetcode.com/problems/evaluate-division/
# Companies: Amazon,Google,Bloomberg
# Categories: Union Find,Graph

# ----------------------------------------------------------------------------

class Solution(object):
    def calcEquation(self, equations, values, queries):
        """
        :type equations: List[List[str]]
        :type values: List[float]
        :type queries: List[List[str]]
        :rtype: List[float]
        """
        graph = {}
        for i, (start, end) in enumerate(equations):
            graph.setdefault(start, {})[end] = float(values[i])
            graph.setdefault(end, {})[start] = 1 / float(values[i])

        def dist(start, end):
            if start not in graph: return -1
            bfs = [(start, 1)]
            visited = set()
            while bfs:
                end_node = next((val for node, val in bfs if node == end), None)
                if end_node is not None: return end_node
                bfs = [ (to_node, cur_val * graph[cur_node][to_node])
                            for cur_node, cur_val in bfs
                                for to_node in graph[cur_node]
                                    if to_node not in visited and (visited.add(to_node) is None)]
            return -1
        return [dist(start, end) for start, end in queries]


# ============================================================================

# 401. Binary Watch
# Difficulty: Easy
# link: https://leetcode.com/problems/binary-watch/
# Companies: Apple
# Categories: Backtracking,Bit Manipulation

# ----------------------------------------------------------------------------

class Solution(object):
    def readBinaryWatch(self, num):
        """
        :type num: int
        :rtype: List[str]
        """
        def count_ones(num):
            count = 0
            while num != 0:
                count += num % 2
                num /= 2
            return count
        return [
            "%d:%02d" %(hr, m)
            for hr in range(12)
            for m in range(60)
            if count_ones(hr) + count_ones(m) == num
        ]


# ============================================================================

# 403. Frog Jump
# Difficulty: Hard
# link: https://leetcode.com/problems/frog-jump/
# Companies: Oracle,Amazon,Google,Nutanix
# Categories: Dynamic Programming

# ----------------------------------------------------------------------------

class Solution(object):
    def canCross(self, stones):
        """
        :type stones: List[int]
        :rtype: bool
        """
        from collections import defaultdict
        stone_to_steps = defaultdict(set)
        if (stones[1] - stones[0]) != 1: return False
        stone_to_steps[stones[1]].add(1)
        for pos in stones:
            for step in stone_to_steps[pos]:
                for new_pos in [pos + step + i for i in [-1, 0, 1]]:
                    if new_pos == stones[-1]: return True
                    elif new_pos != pos:
                        stone_to_steps[new_pos].add(new_pos - pos)
        return False


# ============================================================================

# 404. Sum of Left Leaves
# Difficulty: Easy
# link: https://leetcode.com/problems/sum-of-left-leaves/
# Companies: Mathworks,Amazon,Google,Expedia
# Categories: Tree

# ----------------------------------------------------------------------------

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def sumOfLeftLeaves(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        if not root: return 0
        left_leave_sum = 0
        bfs = [root]
        while bfs:
            left_leave_sum += sum(node.left.val for node in bfs if node.left and not node.left.right and not node.left.left)
            bfs = [kid for node in bfs for kid in [node.left, node.right] if kid]
        return left_leave_sum


# ============================================================================

# 405. Convert a Number to Hexadecimal
# Difficulty: Easy
# link: https://leetcode.com/problems/convert-a-number-to-hexadecimal/
# Companies: Facebook
# Categories: Bit Manipulation

# ----------------------------------------------------------------------------

class Solution(object):
    def toHex(self, num):
        """
        :type num: int
        :rtype: str
        """
        if not num: return '0'
        res = ''
        mask = reduce(lambda x,y: x | (1 << y), range(4), 0)
        mapping = '0123456789abcdef'
        for _ in range(8):
            res = mapping[(num & mask) % len(mapping)] + res
            num >>= 4
        return res.lstrip('0')


# ============================================================================

# 406. Queue Reconstruction by Height
# Difficulty: Medium
# link: https://leetcode.com/problems/queue-reconstruction-by-height/
# Companies: Google,Microsoft
# Categories: Greedy

# ----------------------------------------------------------------------------

class Solution(object):
    def reconstructQueue(self, people):
        """
        :type people: List[List[int]]
        :rtype: List[List[int]]
        """
        people.sort(key=lambda(h, k): (-h, k))
        res = []
        for p in people:
            res.insert(p[1], p)
        return res


# ============================================================================

# 409. Longest Palindrome
# Difficulty: Easy
# link: https://leetcode.com/problems/longest-palindrome/
# Companies: Amazon
# Categories: Hash Table

# ----------------------------------------------------------------------------

class Solution(object):
    def longestPalindrome(self, s):
        """
        :type s: str
        :rtype: int
        """
        from collections import Counter
        odds_len = sum(count & 1 for count in Counter(s).values())
        return len(s) - odds_len + bool(odds_len)


# ============================================================================

# 412. Fizz Buzz
# Difficulty: Easy
# link: https://leetcode.com/problems/fizz-buzz/
# Companies: Amazon,Capital One,Apple,LinkedIn,Microsoft
# Categories:

# ----------------------------------------------------------------------------

class Solution(object):
    def fizzBuzz(self, n):
        """
        :type n: int
        :rtype: List[str]
        """
        return [('Fizz' * (not i % 3) + 'Buzz' * (not i % 5)) or str(i) for i in xrange(1, n + 1)]


# ============================================================================

# 413. Arithmetic Slices
# Difficulty: Medium
# link: https://leetcode.com/problems/arithmetic-slices/
# Companies: Baidu,Amazon,Facebook,Aetion,Bloomberg
# Categories: Math,Dynamic Programming

# ----------------------------------------------------------------------------

class Solution(object):
    def numberOfArithmeticSlices(self, A):
        """
        :type A: List[int]
        :rtype: int
        """
        diffs = []
        prev = None
        for i in xrange(1, len(A)):
            diff = A[i] - A[i - 1]
            if not diffs or prev != diff:
                diffs.append(1)
            else: diffs[-1] += 1
            prev = diff
        return sum((n * (n - 1) / 2) for n in diffs)


# ============================================================================

# 414. Third Maximum Number
# Difficulty: Easy
# link: https://leetcode.com/problems/third-maximum-number/
# Companies: Facebook
# Categories: Array

# ----------------------------------------------------------------------------

class Solution(object):
    def thirdMax(self, nums):
        import heapq
        if not nums: return []
        else:
            nums = list(set(nums))
            if len(nums) < 3: return max(nums)
            for _ in range(2):
                nums[nums.index(max(nums))] = float('-inf')
            return max(nums)


# ============================================================================

# 415. Add Strings
# Difficulty: Easy
# link: https://leetcode.com/problems/add-strings/
# Companies: Adobe,Facebook,Microsoft,Snapchat
# Categories: String

# ----------------------------------------------------------------------------

class Solution(object):
    def addStrings(self, num1, num2):
        """
        :type num1: str
        :type num2: str
        :rtype: str
        """
        def convert_to_int(num):
            res = 0
            for digit in num:
                res *= 10
                res += ord(digit) - ord('0')
            return res
        return str(convert_to_int(num1) + convert_to_int(num2))


# ============================================================================

# 416. Partition Equal Subset Sum
# Difficulty: Medium
# link: https://leetcode.com/problems/partition-equal-subset-sum/
# Companies: Amazon,Google,Facebook
# Categories: Dynamic Programming

# ----------------------------------------------------------------------------

class Solution(object):
    def canPartition(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        target = sum(nums) / 2.
        all_sum = {0}
        for num in nums: all_sum |= {num + i for i in all_sum}
        return target in all_sum


# ============================================================================

# 419. Battleships in a Board
# Difficulty: Medium
# link: https://leetcode.com/problems/battleships-in-a-board/
# Companies: Microsoft,Apple
# Categories:

# ----------------------------------------------------------------------------

class Solution(object):
    def countBattleships(self, board):
        """
        :type board: List[List[str]]
        :rtype: int
        """
        m, n = len(board), len(board[0])
        for i in range(m):
            for j in range(n):
                if i + 1 < m and 'X' == board[i + 1][j] or \
                    j + 1 < n and 'X' == board[i][j + 1]:
                    board[i][j] = '.'
        return sum(1 for row in board for el in row if el == 'X')


# ============================================================================

# 421. Maximum XOR of Two Numbers in an Array
# Difficulty: Medium
# link: https://leetcode.com/problems/maximum-xor-of-two-numbers-in-an-array/
# Companies: Google
# Categories: Bit Manipulation,Trie

# ----------------------------------------------------------------------------

class Solution(object):
    def findMaximumXOR(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        res = 0
        for i in reversed(range(32)):
            prefixes = set(x >> i for x in nums)
            res <<= 1
            res += any((res+1) ^ p in prefixes for p in prefixes)
        return res


# ============================================================================

# 423. Reconstruct Original Digits from English
# Difficulty: Medium
# link: https://leetcode.com/problems/reconstruct-original-digits-from-english/
# Companies:
# Categories: Math

# ----------------------------------------------------------------------------

class Solution(object):
    def originalDigits(self, s):
        '''
        zero: Only digit with z
        two: Only digit with w
        four: Only digit with u
        six: Only digit with x
        eight: Only digit with g
        '''
        from collections import Counter
        s = Counter(s)
        idx_to_word = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
        def get_count(char, num, counts):
            counts[num] = s.get(char, 0) - sum(counts[i] for i, word in enumerate(idx_to_word) if char in word and i != num)

        counts = [
            s.get('z', 0), None,
            s.get('w', 0), None,
            s.get('u', 0), None,
            s.get('x', 0), None,
            s.get('g', 0), None,
        ]
        get_count('o', 1, counts)
        get_count('t', 3, counts)
        get_count('f', 5, counts)
        get_count('s', 7, counts)
        get_count('i', 9, counts)
        return ''.join(str(i) * occ for i, occ in enumerate(counts))


# ============================================================================

# 424. Longest Repeating Character Replacement
# Difficulty: Medium
# link: https://leetcode.com/problems/longest-repeating-character-replacement/
# Companies: Google,Pocket Gems
# Categories: Two Pointers,Sliding Window

# ----------------------------------------------------------------------------

class Solution(object):
    def characterReplacement(self, s, k):
        """
        :type s: str
        :type k: int
        :rtype: int
        """
        counts = {}
        max_len = start_i = 0
        for end_i, end in enumerate(s):
            counts[end] = counts.get(end, 0) + 1
            max_len = max(max_len, counts[end])
            while ((end_i - start_i + 1) - max_len) > k:
                counts[s[start_i]] -= 1
                start_i += 1
        return max_len + min(k, len(s) - max_len)


# ============================================================================

# 434. Number of Segments in a String
# Difficulty: Easy
# link: https://leetcode.com/problems/number-of-segments-in-a-string/
# Companies:
# Categories: String

# ----------------------------------------------------------------------------

class Solution(object):
    def countSegments(self, s):
        if not s: return 0
        return sum(s[i - 1] == " " and s[i] != " " for i in range(1, len(s))) + (s[0] != " ")


# ============================================================================

# 435. Non-overlapping Intervals
# Difficulty: Medium
# link: https://leetcode.com/problems/non-overlapping-intervals/
# Companies: Amazon
# Categories: Greedy

# ----------------------------------------------------------------------------

# Definition for an interval.
# class Interval(object):
#     def __init__(self, s=0, e=0):
#         self.start = s
#         self.end = e

class Solution(object):
    def eraseOverlapIntervals(self, intervals):
        """
        :type intervals: List[Interval]
        :rtype: int
        """
        def not_overlap(int1, int2):
            return min(int1.end, int2.end) <= max(int1.start, int2.start)
        def eft_cmp(x, y):
            if x.end < y.end or x.start < y.start:
                return -1
            elif x.end > y.end or x.start > y.start:
                return 1
            else:
                return 0
        intervals.sort(eft_cmp)
        count = 0
        prev = None
        for interval in intervals:
            if (prev and not_overlap(prev, interval)) or not prev :
                count += 1
                prev = interval
        return len(intervals) - count


# ============================================================================

# 436. Find Right Interval
# Difficulty: Medium
# link: https://leetcode.com/problems/find-right-interval/
# Companies:
# Categories: Binary Search

# ----------------------------------------------------------------------------

# Definition for an interval.
# class Interval(object):
#     def __init__(self, s=0, e=0):
#         self.start = s
#         self.end = e

class Solution(object):
    def findRightInterval(self, intervals):
        """
        :type intervals: List[Interval]
        :rtype: List[int]
        """
        from bisect import bisect_left
        start_idx = sorted([i.start, idx] for idx, i in enumerate(intervals)) + [[float('inf'), -1]]
        return [start_idx[bisect_left(start_idx, [i.end])][1] for i in intervals]


# ============================================================================

# 437. Path Sum III
# Difficulty: Easy
# link: https://leetcode.com/problems/path-sum-iii/
# Companies: Amazon,Quora
# Categories: Tree

# ----------------------------------------------------------------------------

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def pathSum(self, root, sum):
        """
        :type root: TreeNode
        :type sum: int
        :rtype: int
        """
        def _pathSum(node, target, sums_count={0: 1}, so_far=0):
            if not node: return 0
            so_far += node.val
            count = sums_count.get(so_far - target, 0)
            sums_count.setdefault(so_far, 0)
            sums_count[so_far] += 1
            count += _pathSum(node.left, target, sums_count, so_far)
            count += _pathSum(node.right, target, sums_count, so_far)
            sums_count[so_far] -= 1
            if so_far in sums_count and not sums_count[so_far]: del sums_count[so_far]
            return count
        return _pathSum(root, sum)


# ============================================================================

# 438. Find All Anagrams in a String
# Difficulty: Medium
# link: https://leetcode.com/problems/find-all-anagrams-in-a-string/
# Companies: Amazon,Facebook
# Categories: Hash Table

# ----------------------------------------------------------------------------

class Solution(object):
    def findAnagrams(self, s, p):

        if len(s) < len(p): return []

        from collections import Counter
        p_cnt = Counter(p)
        self.cnt = 0
        res = []

        def update_val(idx, val):
            if s[idx] in p_cnt:
                prev = p_cnt[s[idx]]
                p_cnt[s[idx]] += val
                if prev == 0: self.cnt -= 1
                elif prev != 0 and p_cnt[s[idx]] == 0: self.cnt += 1

        for i in range(len(s)):
            if i < len(p): update_val(i, -1)
            else:
                update_val(i - len(p), 1)
                update_val(i, -1)
            if self.cnt == len(p_cnt): res.append(i - len(p) + 1)
        return res


# ============================================================================

# 442. Find All Duplicates in an Array
# Difficulty: Medium
# link: https://leetcode.com/problems/find-all-duplicates-in-an-array/
# Companies: Amazon,Lyft
# Categories: Array

# ----------------------------------------------------------------------------

class Solution(object):
    def findDuplicates(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        res = []
        for i in xrange(len(nums)):
            idx = abs(nums[i]) - 1
            if nums[idx] < 0: res.append(idx + 1)
            else: nums[idx] = -nums[idx]
        return res


# ============================================================================

# 443. String Compression
# Difficulty: Easy
# link: https://leetcode.com/problems/string-compression/
# Companies: Expedia,Apple,Wayfair,Amazon,Google,Microsoft
# Categories: String

# ----------------------------------------------------------------------------

class Solution(object):
    def compress(self, chars):
        i = j = cnt = 0
        chars.append('blah')
        while j < len(chars):
            if j == 0 or chars[j] == chars[j - 1]:
                cnt, j = cnt + 1, j + 1
            elif chars[j] != chars[j - 1]:
                chars[i] = str(chars[j - 1])
                if cnt > 1:
                    for char in str(cnt):
                        i += 1
                        chars[i] = char
                i += 1
                j += 1
                cnt = 1
        # while len(chars) != i: chars.pop()
        return i


# ============================================================================

# 444. Sequence Reconstruction
# Difficulty: Medium
# link: https://leetcode.com/problems/sequence-reconstruction/
# Companies: Google
# Categories: Graph,Topological Sort

# ----------------------------------------------------------------------------

class Solution(object):
    def sequenceReconstruction(self, org, seqs):

        from collections import defaultdict
        G = defaultdict(set)
        for seq in seqs:
            if len(seq): G[seq[0]]
            for i in range(1, len(seq)):
                G[seq[i - 1]].add(seq[i])

        def dfs(node, res = [], visited = set()):
            while G[node]:
                adj = G[node].pop()
                if adj not in visited:
                    dfs(adj, res)
                    visited.add(adj)
            res.append(node)
            return res

        return org[0] in G and dfs(org[0])[::-1] == org and len(set(G.keys()))==len(org)


# ============================================================================

# 445. Add Two Numbers II
# Difficulty: Medium
# link: https://leetcode.com/problems/add-two-numbers-ii/
# Companies: Amazon,Microsoft,Bloomberg
# Categories: Linked List

# ----------------------------------------------------------------------------

# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def addTwoNumbers(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """

        def _get_stack(node):
            stack = []
            while node:
                stack.append(node.val)
                node = node.next
            return stack
        s1 = _get_stack(l1)
        s2 = _get_stack(l2)

        carry = 0
        dummy = ListNode('dummy')
        while s1 or s2 or carry:
            cur_val = carry
            if s1:
                cur_val += s1.pop()
            if s2:
                cur_val += s2.pop()
            carry, cur_val = cur_val/10, cur_val%10
            cur_node = ListNode(cur_val)
            cur_node.next, dummy.next = dummy.next, cur_node
        return dummy.next


# ============================================================================

# 447. Number of Boomerangs
# Difficulty: Easy
# link: https://leetcode.com/problems/number-of-boomerangs/
# Companies: Google
# Categories: Hash Table

# ----------------------------------------------------------------------------

class Solution(object):
    def numberOfBoomerangs(self, points):
        """
        :type points: List[List[int]]
        :rtype: int
        """
        res = 0
        for i, [x, y] in enumerate(points):
            dist_to_point = {}
            for j, [adj_x, adj_y] in enumerate(points):
                if i != j:
                    key = (x - adj_x) ** 2 + (y - adj_y) ** 2
                    dist_to_point[key] = dist_to_point.setdefault(key, 0) + 1
            res += sum([val * (val - 1) for val in dist_to_point.values()])
        return res


# ============================================================================

# 448. Find All Numbers Disappeared in an Array
# Difficulty: Easy
# link: https://leetcode.com/problems/find-all-numbers-disappeared-in-an-array/
# Companies: Google,Apple
# Categories: Array

# ----------------------------------------------------------------------------

class Solution(object):
    def findDisappearedNumbers(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        for num in nums: nums[abs(num) - 1] = -1 * abs(nums[abs(num) - 1])
        return [i + 1 for i, num in enumerate(nums) if num > 0]


# ============================================================================

# 449. Serialize and Deserialize BST
# Difficulty: Medium
# link: https://leetcode.com/problems/serialize-and-deserialize-bst/
# Companies: Uber,Amazon,Facebook
# Categories: Tree

# ----------------------------------------------------------------------------

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Codec:

    def serialize(self, root):
        """Encodes a tree to a single string.

        :type root: TreeNode
        :rtype: str
        """
        def _serialize(node):
            if not node: return
            return (node.val, _serialize(node.left), _serialize(node.right))
        return str(_serialize(root))

    def deserialize(self, data):
        """Decodes your encoded data to tree.

        :type data: str
        :rtype: TreeNode
        """
        def _deserialize(input_tuple):
            if not input_tuple: return
            node = TreeNode(input_tuple[0])
            node.left = _deserialize(input_tuple[1])
            node.right = _deserialize(input_tuple[2])
            return node

        return  _deserialize(eval(data))


# Your Codec object will be instantiated and called as such:
# codec = Codec()
# codec.deserialize(codec.serialize(root))


# ============================================================================

# 450. Delete Node in a BST
# Difficulty: Medium
# link: https://leetcode.com/problems/delete-node-in-a-bst/
# Companies: Microsoft
# Categories: Tree

# ----------------------------------------------------------------------------

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def deleteNode(self, root, key):
        """
        :type root: TreeNode
        :type key: int
        :rtype: TreeNode
        """
        def _findMinNode(node):
            node = node.right
            while node.left:
                node = node.left
            return node

        def _deleteNode(root, key):
            if not root:
                return
            if root.val > key:
                root.left = _deleteNode(root.left, key)
            elif root.val < key:
                root.right = _deleteNode(root.right, key)
            else:
                if not (root.left and root.right):
                    if root.left:
                        return root.left
                    elif root.right:
                        return root.right
                    else:
                        return
                else:
                    min_node = _findMinNode(root)
                    root.val = min_node.val
                    root.right = _deleteNode(root.right, min_node.val)
            return root
        return _deleteNode(root, key)


# ============================================================================

# 451. Sort Characters By Frequency
# Difficulty: Medium
# link: https://leetcode.com/problems/sort-characters-by-frequency/
# Companies: Google,Bloomberg
# Categories: Hash Table,Heap

# ----------------------------------------------------------------------------

class Solution(object):
    def frequencySort(self, s):
        """
        :type s: str
        :rtype: str
        """
        from collections import Counter
        counts = Counter(s)
        counts = [(freq, char) for char, freq in counts.iteritems()]
        counts.sort(reverse=True)
        for i in range(len(counts)): counts[i] = counts[i][1] * counts[i][0]
        return ''.join(counts)


# ============================================================================

# 452. Minimum Number of Arrows to Burst Balloons
# Difficulty: Medium
# link: https://leetcode.com/problems/minimum-number-of-arrows-to-burst-balloons/
# Companies: Amazon,Facebook
# Categories: Greedy

# ----------------------------------------------------------------------------

class Solution(object):
    def findMinArrowShots(self, points):
        points.sort()
        last_start = None
        cnt = 0
        for i in range(len(points) - 1, -1, -1):
            if last_start is None or points[i][1] < last_start:
                cnt += 1
                last_start = points[i][0]
        return cnt


# ============================================================================

# 453. Minimum Moves to Equal Array Elements
# Difficulty: Easy
# link: https://leetcode.com/problems/minimum-moves-to-equal-array-elements/
# Companies: Drawbridge
# Categories: Math

# ----------------------------------------------------------------------------

class Solution(object):
    def minMoves(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        return sum(nums) - len(nums) * min(nums)


# ============================================================================

# 454. 4Sum II
# Difficulty: Medium
# link: https://leetcode.com/problems/4sum-ii/
# Companies: Amazon
# Categories: Hash Table,Binary Search

# ----------------------------------------------------------------------------

class Solution(object):
    def fourSumCount(self, A, B, C, D):
        from collections import Counter, defaultdict
        sum_count = defaultdict(int, Counter([a + b for a in A for b in B]))
        return sum(sum_count[-d-c] for c in C for d in D)


# ============================================================================

# 455. Assign Cookies
# Difficulty: Easy
# link: https://leetcode.com/problems/assign-cookies/
# Companies:
# Categories: Greedy

# ----------------------------------------------------------------------------

class Solution(object):
    def findContentChildren(self, g, s):
        """
        :type g: List[int]
        :type s: List[int]
        :rtype: int
        """
        g.sort(reverse=True)
        s.sort(reverse=True)
        count = 0
        while s and g:
            cookie = s.pop()
            if cookie >= g[-1]:
                g.pop()
                count += 1
        return count


# ============================================================================

# 456. 132 Pattern
# Difficulty: Medium
# link: https://leetcode.com/problems/132-pattern/
# Companies: Amazon
# Categories: Stack

# ----------------------------------------------------------------------------

class Solution(object):
    def find132pattern(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        s3 = float('-inf')
        stack = []
        for num in reversed(nums):
            if num < s3: return True
            else:
                while stack and stack[-1] < num: s3 = stack.pop()
                stack.append(num)
        return False


# ============================================================================

# 459. Repeated Substring Pattern
# Difficulty: Easy
# link: https://leetcode.com/problems/repeated-substring-pattern/
# Companies: Google,Microsoft
# Categories: String

# ----------------------------------------------------------------------------

class Solution(object):
    def repeatedSubstringPattern(self, s):
        """
        :type s: str
        :rtype: bool
        """
        return s in (s + s)[1:-1]


# ============================================================================

# 461. Hamming Distance
# Difficulty: Easy
# link: https://leetcode.com/problems/hamming-distance/
# Companies: Facebook,Adobe
# Categories: Bit Manipulation

# ----------------------------------------------------------------------------

class Solution(object):
    def hammingDistance(self, x, y):
        count = 0
        while x or y:
            if (x & 1) != (y & 1): count += 1
            x >>= 1
            y >>= 1
        return count


# ============================================================================

# 462. Minimum Moves to Equal Array Elements II
# Difficulty: Medium
# link: https://leetcode.com/problems/minimum-moves-to-equal-array-elements-ii/
# Companies:
# Categories: Math

# ----------------------------------------------------------------------------

class Solution(object):
    def minMoves2(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        nums.sort()
        mid_val = nums[len(nums) / 2]
        return sum(abs(x - mid_val) for x in nums)


# ============================================================================

# 463. Island Perimeter
# Difficulty: Easy
# link: https://leetcode.com/problems/island-perimeter/
# Companies: Amazon,Facebook
# Categories: Hash Table

# ----------------------------------------------------------------------------

class Solution(object):
    def islandPerimeter(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        queue = []
        count = 0
        for i in xrange(len(grid)):
            for j in xrange(len(grid[0])):
                if grid[i][j]:
                    adjs = [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]
                    for adj in adjs:
                        if not (0 <= adj[0] < len(grid) and 0 <= adj[1] < len(grid[0])) or \
                            not grid[adj[0]][adj[1]]:
                            count += 1
        return count


# ============================================================================

# 468. Validate IP Address
# Difficulty: Medium
# link: https://leetcode.com/problems/validate-ip-address/
# Companies: Amazon,Facebook
# Categories: String

# ----------------------------------------------------------------------------

class Solution(object):
    def validIPAddress(self, IP):
        """
        :type IP: str
        :rtype: str
        """
        def is_hex(s):
            hex_digits = set("0123456789abcdefABCDEF")
            for char in s:
                if not (char in hex_digits):
                    return False
            return True
        if '.' in IP:
            IP = IP.split('.')
            if len(IP) != 4:
                return "Neither"
            for ip in IP:
                try:
                    ip_int = int(ip)
                    if ip_int > 255 or ip_int < 0 or str(ip_int) != ip:
                        return "Neither"
                except:
                    return "Neither"
            return 'IPv4'
        elif ':' in IP:
            IP = IP.split(':')
            if len(IP) != 8:
                return "Neither"
            for ip in IP:
                if len(ip) > 4 or len(ip) == 0 or not is_hex(ip):
                    return 'Neither'
            return 'IPv6'
        return "Neither"


# ============================================================================

# 473. Matchsticks to Square
# Difficulty: Medium
# link: https://leetcode.com/problems/matchsticks-to-square/
# Companies: Rackspace
# Categories: Depth-first Search

# ----------------------------------------------------------------------------

class Solution(object):

    def makesquare(self, nums):
        sum_of_elems = sum(nums)
        if len(nums) < 4 or sum_of_elems % 4: return False
        nums.sort(reverse=True)
        def _makesquare(pos, sums):
            if pos >= len(nums): return not any(sums)
            next_elem = nums[pos]
            visited = set()
            for i in range(len(sums)):
                if sums[i] - next_elem >= 0 and sums[i] not in visited:
                    sums[i] -= next_elem
                    if _makesquare(pos + 1, sums): return True
                    sums[i] += next_elem
                    visited.add(sums[i])
            return False
        return _makesquare(0, [sum_of_elems / 4 for _ in range(4)])


# ============================================================================

# 475. Heaters
# Difficulty: Easy
# link: https://leetcode.com/problems/heaters/
# Companies: Google
# Categories: Binary Search

# ----------------------------------------------------------------------------

class Solution(object):
    def findRadius(self, houses, heaters):
        """
        :type houses: List[int]
        :type heaters: List[int]
        :rtype: int
        """
        heaters.sort()
        heaters.append(float('inf'))
        cur = diff = 0
        for house in sorted(houses):
            while cur + 1 < len(heaters) and heaters[cur + 1] < house: cur += 1
            diff = max(diff, min(abs(heaters[cur] - house), abs(heaters[cur + 1] - house)))
        return diff


# ============================================================================

# 476. Number Complement
# Difficulty: Easy
# link: https://leetcode.com/problems/number-complement/
# Companies: Cloudera
# Categories: Bit Manipulation

# ----------------------------------------------------------------------------

class Solution(object):
    def findComplement(self, num):
        """
        :type num: int
        :rtype: int
        """
        shift, x = 0, num
        while x:
            shift += 1
            x >>= 1

        mask = (1 << (shift)) - 1
        return (num ^ mask)


# ============================================================================

# 477. Total Hamming Distance
# Difficulty: Medium
# link: https://leetcode.com/problems/total-hamming-distance/
# Companies: Facebook
# Categories: Bit Manipulation

# ----------------------------------------------------------------------------

class Solution(object):
    def totalHammingDistance(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        nums = [[int(bool(num & (1 << i))) for i in range(31, -1, -1)] for num in nums]
        counts = [sum([num[i] for num in nums]) for i in range(31, -1, -1)]
        return sum([count * (len(nums) - count) for count in counts])


# ============================================================================

# 482. License Key Formatting
# Difficulty: Easy
# link: https://leetcode.com/problems/license-key-formatting/
# Companies: Google
# Categories:

# ----------------------------------------------------------------------------

class Solution(object):
    def licenseKeyFormatting(self, S, K):
        """
        :type S: str
        :type K: int
        :rtype: str
        """
        S = S.replace('-', '').upper()
        return '-'.join([ S[max(len(S) - i, 0) : len(S) - i + K]
                         for i in range(K, len(S) + K, K)][::-1])


# ============================================================================

# 485. Max Consecutive Ones
# Difficulty: Easy
# link: https://leetcode.com/problems/max-consecutive-ones/
# Companies: Amazon
# Categories: Array

# ----------------------------------------------------------------------------

class Solution(object):
    def findMaxConsecutiveOnes(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        max_len = 0
        for j, val in enumerate(nums):
            if val:
                if j - 1 < 0 or not (nums[j - 1]): i = j
                max_len = max(max_len, j - i + 1)
        return max_len


# ============================================================================

# 486. Predict the Winner
# Difficulty: Medium
# link: https://leetcode.com/problems/predict-the-winner/
# Companies: Google
# Categories: Dynamic Programming,Minimax

# ----------------------------------------------------------------------------

class Solution(object):
    def PredictTheWinner(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        dp = [[None] * (len(nums) + 1) for _ in range(len(nums) + 1)]
        def _PredictTheWinner(i, j):
            if dp[i][j] is not None:
                return dp[i][j]
            if i == j:
                dp[i][j] = nums[i], 0
            else:
                o_r, s_r = _PredictTheWinner(i, j - 1)
                o_l, s_l = _PredictTheWinner(i + 1, j)
                s_r += nums[j]
                s_l += nums[i]
                if s_r - o_r > s_l - o_l:
                    dp[i][j] =  s_r, o_r
                else:
                    dp[i][j] = s_l, o_l
            return dp[i][j]
        p1, p2 = _PredictTheWinner(0, len(nums) - 1)
        return p1 >= p2


# ============================================================================

# 491. Increasing Subsequences
# Difficulty: Medium
# link: https://leetcode.com/problems/increasing-subsequences/
# Companies: Facebook,Yahoo
# Categories: Depth-first Search

# ----------------------------------------------------------------------------

class Solution(object):
    def findSubsequences(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        res = {()}
        for num in nums:
            res |= { ary + (num, ) for ary in res if not ary or ary[-1] <= num }
        return [x for x in res if len(x) >= 2]


# ============================================================================

# 492. Construct the Rectangle
# Difficulty: Easy
# link: https://leetcode.com/problems/construct-the-rectangle/
# Companies:
# Categories:

# ----------------------------------------------------------------------------

class Solution(object):
    def constructRectangle(self, area):
        """
        :type area: int
        :rtype: List[int]
        """
        i = j = int(area ** 0.5)
        while (i * j) != area:
            if i * j > area: i -= 1
            else: j += 1
        return j, i


# ============================================================================

# 494. Target Sum
# Difficulty: Medium
# link: https://leetcode.com/problems/target-sum/
# Companies: Facebook
# Categories: Dynamic Programming,Depth-first Search

# ----------------------------------------------------------------------------

class Solution(object):
    def findTargetSumWays(self, nums, S):
        """
        :type nums: List[int]
        :type S: int
        :rtype: int
        """
        dp = {0:1}
        for num in nums:
            new_dp = {}
            for key, val in dp.iteritems():
                for new_key in [key + num, key - num]:
                    new_dp.setdefault(new_key, 0)
                    new_dp[new_key] += val
            dp = new_dp
        return dp.get(S, 0)


# ============================================================================

# 495. Teemo Attacking
# Difficulty: Medium
# link: https://leetcode.com/problems/teemo-attacking/
# Companies: Riot Games
# Categories: Array

# ----------------------------------------------------------------------------

class Solution(object):
    def findPoisonedDuration(self, timeSeries, duration):
        """
        :type timeSeries: List[int]
        :type duration: int
        :rtype: int
        """
        total_time = 0
        for i, time in enumerate(timeSeries):
            total_time += duration
            time_diff = timeSeries[i] - timeSeries[i-1]
            if i > 0 and time_diff < duration:
                total_time -= duration - time_diff
        return total_time


# ============================================================================

# 496. Next Greater Element I
# Difficulty: Easy
# link: https://leetcode.com/problems/next-greater-element-i/
# Companies: Amazon,Facebook,Twitter
# Categories: Stack

# ----------------------------------------------------------------------------

class Solution(object):
    def nextGreaterElement(self, findNums, nums):
        """
        :type findNums: List[int]
        :type nums: List[int]
        :rtype: List[int]
        """
        return [next((num for num in nums[nums.index(num_f):] if num > num_f), -1) for num_f in findNums]


# ============================================================================

# 498. Diagonal Traverse
# Difficulty: Medium
# link: https://leetcode.com/problems/diagonal-traverse/
# Companies: Facebook,Amazon,Google,Walmart Labs,eBay
# Categories:

# ----------------------------------------------------------------------------

class Solution(object):
    def findDiagonalOrder(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: List[int]
        """
        if not matrix or not all(matrix): return []
        res = []
        def get_diag(i, j):
            new_diag = []
            while i >= 0 and j < len(matrix[0]):
                new_diag.append(matrix[i][j])
                i -= 1
                j += 1
            res.append(new_diag)
        for i in range(len(matrix)): get_diag(i, 0)
        for j in range(1, len(matrix[0])): get_diag(len(matrix) - 1, j)
        for i in range(1, len(res), 2): res[i] = list((reversed(res[i])))
        return [item for lst in res for item in lst]


# ============================================================================

# 500. Keyboard Row
# Difficulty: Easy
# link: https://leetcode.com/problems/keyboard-row/
# Companies: Mathworks
# Categories: Hash Table

# ----------------------------------------------------------------------------

class Solution(object):
    def findWords(self, words):
        """
        :type words: List[str]
        :rtype: List[str]
        """
        keyboard = ['qwertyuiop', 'asdfghjkl', 'zxcvbnm']
        res = []
        for word in words:
            char = word[0].lower()
            for row in keyboard:
                if char in row:
                    if all(char.lower() in row for char in word):
                        res.append(word)
                        break
        return res


# ============================================================================

# 501. Find Mode in Binary Search Tree
# Difficulty: Easy
# link: https://leetcode.com/problems/find-mode-in-binary-search-tree/
# Companies: Google
# Categories: Tree

# ----------------------------------------------------------------------------

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def findMode(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        counts = {}
        if not root: return []
        bfs = [root]
        while bfs:
            for node in bfs:
                counts[node.val] = counts.get(node.val, 0) + 1
            bfs = [kid for node in bfs for kid in [node.left, node.right] if kid]
        max_freq = max(counts.values())
        return [key for key, freq in counts.iteritems() if freq == max_freq]


# ============================================================================

# 503. Next Greater Element II
# Difficulty: Medium
# link: https://leetcode.com/problems/next-greater-element-ii/
# Companies: Amazon,Bloomberg
# Categories: Stack

# ----------------------------------------------------------------------------

class Solution(object):
    def nextGreaterElements(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        stack, res = [], [-1] * len(nums)
        for i in range(len(nums)) * 2:
            while stack and nums[stack[-1]] < nums[i]:
                res[stack.pop()] = nums[i]
            stack.append(i)
        return res


# ============================================================================

# 504. Base 7
# Difficulty: Easy
# link: https://leetcode.com/problems/base-7/
# Companies: Google
# Categories:

# ----------------------------------------------------------------------------

class Solution(object):
    def convertToBase7(self, num):
        """
        :type num: int
        :rtype: str
        """
        base = 7
        res = ""
        neg_prefix = '-' if num < 0 else ''
        num = abs(num)
        while num != 0:
            num, mod = divmod(num, base)
            res = str(mod) + res
        return (neg_prefix + res) or '0'


# ============================================================================

# 506. Relative Ranks
# Difficulty: Easy
# link: https://leetcode.com/problems/relative-ranks/
# Companies: Google
# Categories:

# ----------------------------------------------------------------------------

class Solution(object):
    def findRelativeRanks(self, nums):
        """
        :type nums: List[int]
        :rtype: List[str]
        """
        sorted_scores = sorted(nums, reverse=True)
        str_scores = ["Gold Medal", "Silver Medal", "Bronze Medal"]
        str_scores = (str_scores + [str(i) for i in xrange(4, len(nums) + 1)])[:len(nums)]
        score_to_str = dict(zip(sorted_scores, str_scores))
        return [score_to_str[score] for score in nums]


# ============================================================================

# 507. Perfect Number
# Difficulty: Easy
# link: https://leetcode.com/problems/perfect-number/
# Companies: Amazon
# Categories: Math

# ----------------------------------------------------------------------------

class Solution(object):
    def checkPerfectNumber(self, num):
        """
        :type num: int
        :rtype: bool
        """
        if num <= 0: return False
        sqrt = num ** 0.5
        div_sum = sum(j for i in xrange(2, int(sqrt) + 1) if num % i == 0 for j in [i, num / i]) + 1
        if int(sqrt) == sqrt: div_sum -= sqrt
        return div_sum == num


# ============================================================================

# 508. Most Frequent Subtree Sum
# Difficulty: Medium
# link: https://leetcode.com/problems/most-frequent-subtree-sum/
# Companies: Amazon
# Categories: Hash Table,Tree

# ----------------------------------------------------------------------------

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def findFrequentTreeSum(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        if not root: return []
        from collections import defaultdict
        val_to_freq = defaultdict(lambda:0)
        def _findFrequentTreeSum(node):
            if not node: return 0
            tree_sum = node.val + _findFrequentTreeSum(node.left) + _findFrequentTreeSum(node.right)
            val_to_freq[tree_sum] += 1
            return tree_sum
        _findFrequentTreeSum(root)
        max_freq = max(val_to_freq.values())
        return [val for val, freq in val_to_freq.iteritems() if max_freq == freq]


# ============================================================================

# 509. Fibonacci Number
# Difficulty: Easy
# link: https://leetcode.com/problems/fibonacci-number/
# Companies: Barclays,Microsoft
# Categories: Array

# ----------------------------------------------------------------------------

class Solution(object):
    def fib(self, N):
        """
        :type N: int
        :rtype: int
        """
        if N <= 1: return N
        p, c = 0, 1
        for i in range(N-1): p, c = c, p + c
        return c


# ============================================================================

# 513. Find Bottom Left Tree Value
# Difficulty: Medium
# link: https://leetcode.com/problems/find-bottom-left-tree-value/
# Companies: Microsoft
# Categories: Tree,Depth-first Search,Breadth-first Search

# ----------------------------------------------------------------------------

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def findBottomLeftValue(self, root):
        prev = []
        bfs = [root]
        while bfs:
            prev = bfs
            bfs = [kid for node in bfs for kid in [node.left, node.right] if kid]
        return prev[0].val


# ============================================================================

# 515. Find Largest Value in Each Tree Row
# Difficulty: Medium
# link: https://leetcode.com/problems/find-largest-value-in-each-tree-row/
# Companies: Microsoft
# Categories: Tree,Depth-first Search,Breadth-first Search

# ----------------------------------------------------------------------------

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def largestValues(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        if not root: return []
        bfs = [root]
        res = []
        while bfs:
            res.append(max(map(lambda node: node.val, bfs)))
            bfs = [kid for node in bfs for kid in (node.left, node.right) if kid]
        return res


# ============================================================================

# 518. Coin Change 2
# Difficulty: Medium
# link: https://leetcode.com/problems/coin-change-2/
# Companies: Oracle,Uber,Amazon
# Categories:

# ----------------------------------------------------------------------------

class Solution(object):
    def change(self, amount, coins):
        def cnt(tar=amount, i=0, memo={}):
            if tar > 0 and i < len(coins):
                if (tar, i) in memo: return memo[(tar, i)]
                return memo.setdefault((tar, i), cnt(tar - coins[i], i) + cnt(tar, i + 1))
            return int(tar == 0)
        return cnt()



    def _change(self, amount, coins):
        dp = [0] * (amount + 1)
        dp[0] = 1
        for c in coins:
            for i in range(c, amount + 1):
                dp[i] += dp[i - c]
        return dp[-1]


# ============================================================================

# 520. Detect Capital
# Difficulty: Easy
# link: https://leetcode.com/problems/detect-capital/
# Companies: Google
# Categories: String

# ----------------------------------------------------------------------------

class Solution(object):
    def detectCapitalUse(self, word):
        """
        :type word: str
        :rtype: bool
        """
        is_chars_upper = [(char == char.upper()) for char in reversed(word)]
        first_char_upper = is_chars_upper.pop()
        return first_char_upper and all(is_chars_upper) or not any(is_chars_upper)


# ============================================================================

# 521. Longest Uncommon Subsequence I
# Difficulty: Easy
# link: https://leetcode.com/problems/longest-uncommon-subsequence-i/
# Companies: Google
# Categories: String

# ----------------------------------------------------------------------------

class Solution(object):
    def findLUSlength(self, a, b):
        """
        :type a: str
        :type b: str
        :rtype: int
        """
        return -1 if a == b else max(len(a), len(b))


# ============================================================================

# 524. Longest Word in Dictionary through Deleting
# Difficulty: Medium
# link: https://leetcode.com/problems/longest-word-in-dictionary-through-deleting/
# Companies: Google
# Categories: Two Pointers,Sort

# ----------------------------------------------------------------------------

class Solution(object):
    def findLongestWord(self, s, d):
        """
        :type s: str
        :type d: List[str]
        :rtype: str
        """
        def subseq_str(s, subseq_s):
            subseq_s = list(subseq_s)[::-1]
            for char in s:
                if char == subseq_s[-1]: subseq_s.pop()
                if not subseq_s: return True
            return False
        d.sort(key=lambda word: (-len(word), word))
        return next((word for word in d if subseq_str(s, word)), "")


# ============================================================================

# 525. Contiguous Array
# Difficulty: Medium
# link: https://leetcode.com/problems/contiguous-array/
# Companies: Amazon,Quora,Robinhood
# Categories: Hash Table

# ----------------------------------------------------------------------------

class Solution(object):
    def findMaxLength(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        for i in range(len(nums)): nums[i] = -1 if nums[i] == 0 else 1
        sum_to_idx, sum_so_far, max_len = {0: -1}, 0, 0
        for i in range(len(nums)):
            sum_so_far += nums[i]
            if sum_so_far in sum_to_idx:
                max_len = max(i - sum_to_idx[sum_so_far], max_len)
            sum_to_idx.setdefault(sum_so_far, i)
        return max_len


# ============================================================================

# 529. Minesweeper
# Difficulty: Medium
# link: https://leetcode.com/problems/minesweeper/
# Companies: Uber,Amazon,Google,Cruise Automation,Microsoft
# Categories: Depth-first Search,Breadth-first Search

# ----------------------------------------------------------------------------

class Solution(object):
    def updateBoard(self, board, click):
        def get_adjs(i, j):
            dif = [-1, 0, 1]
            xy_dir = [(x, y) for x in dif for y in dif if x or y]
            adjs = [(i + x, j + y) for x, y in xy_dir]
            adjs = [(x, y) for x, y in adjs if 0 <= x < len(board) and 0 <= y < len(board[0])]
            return adjs
        def click_board(i, j):
            if board[i][j] == 'M':
                board[i][j] = 'X'
            elif board[i][j] == 'E':
                adjs = get_adjs(i, j)
                board[i][j] = str(sum(1 for x, y in adjs if board[x][y] == 'M') or 'B')
                if board[i][j] == 'B':
                    for x, y in adjs: click_board(x, y)
        click_board(*click)
        return board


# ============================================================================

# 530. Minimum Absolute Difference in BST
# Difficulty: Easy
# link: https://leetcode.com/problems/minimum-absolute-difference-in-bst/
# Companies: Apple
# Categories: Tree

# ----------------------------------------------------------------------------

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def getMinimumDifference(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        if not root: return
        bfs = [root]
        vals = []
        while bfs:
            vals.extend([node.val for node in bfs])
            bfs = [kid for node in bfs for kid in [node.left, node.right] if kid]
        vals.sort()
        return min(vals[i] - vals[i - 1] for i in xrange(1, len(vals)))


# ============================================================================

# 532. K-diff Pairs in an Array
# Difficulty: Easy
# link: https://leetcode.com/problems/k-diff-pairs-in-an-array/
# Companies: Twitter,Twilio
# Categories: Array,Two Pointers

# ----------------------------------------------------------------------------

class Solution(object):
    def findPairs(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        from collections import Counter
        counts = Counter(nums)
        res = 0
        for val in counts:
            if k > 0 and val + k in counts or \
                not k and counts[val] > 1:
                res += 1
        return res


# ============================================================================

# 535. Encode and Decode TinyURL
# Difficulty: Medium
# link: https://leetcode.com/problems/encode-and-decode-tinyurl/
# Companies: Amazon,Microsoft
# Categories: Hash Table,Math

# ----------------------------------------------------------------------------

class Codec:

    def __init__(self):
        self.num_to_url = []

    def encode(self, longUrl):
        """Encodes a URL to a shortened URL.

        :type longUrl: str
        :rtype: str
        """
        self.num_to_url.append(longUrl)
        return "http://tinyurl.com/" + str(len(self.num_to_url))

    def decode(self, shortUrl):
        """Decodes a shortened URL to its original URL.

        :type shortUrl: str
        :rtype: str
        """
        return self.num_to_url[int(shortUrl[int(shortUrl.rfind('/')) + 1:]) - 1]

# Your Codec object will be instantiated and called as such:
# codec = Codec()
# codec.decode(codec.encode(url))


# ============================================================================

# 538. Convert BST to Greater Tree
# Difficulty: Easy
# link: https://leetcode.com/problems/convert-bst-to-greater-tree/
# Companies: Amazon
# Categories: Tree

# ----------------------------------------------------------------------------

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def convertBST(self, root):
        """
        :type root: TreeNode
        :rtype: TreeNode
        """
        self.cur_sum = 0
        def _convertBST(node):
            if not node: return
            _convertBST(node.right)
            node.val = self.cur_sum = node.val + self.cur_sum
            _convertBST(node.left)
        _convertBST(root)
        return root


# ============================================================================

# 539. Minimum Time Difference
# Difficulty: Medium
# link: https://leetcode.com/problems/minimum-time-difference/
# Companies: Palantir Technologies
# Categories: String

# ----------------------------------------------------------------------------

class Solution(object):
    def findMinDifference(self, timePoints):
        """
        :type timePoints: List[str]
        :rtype: int
        """
        timePoints = map(lambda x: [int(i) for i in x.split(':')], timePoints)
        timePoints = map(lambda x: x[0] * 60 + x[1], timePoints)
        min_in_a_day = 24*60
        hash_to_bucket = [False] * min_in_a_day
        for time in timePoints:
            if hash_to_bucket[time]:
                return 0
            hash_to_bucket[time] = True
        prev = None
        first = None
        min_diff = float('inf')
        for i, val in enumerate(hash_to_bucket):
            if val:
                if prev is not None:
                    min_diff = min(i - prev, min_diff)
                else:
                    first = i
                prev = i
        return min(min_in_a_day - (prev - first), min_diff)


# ============================================================================

# 540. Single Element in a Sorted Array
# Difficulty: Medium
# link: https://leetcode.com/problems/single-element-in-a-sorted-array/
# Companies: Amazon,Google,Facebook,Microsoft
# Categories:

# ----------------------------------------------------------------------------

class Solution(object):
    def singleNonDuplicate(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        res = 0
        for num in nums: res ^= num
        return res


# ============================================================================

# 541. Reverse String II
# Difficulty: Easy
# link: https://leetcode.com/problems/reverse-string-ii/
# Companies: Amazon
# Categories: String

# ----------------------------------------------------------------------------

class Solution(object):
    def reverseStr(self, s, k):
        """
        :type s: str
        :type k: int
        :rtype: str
        """
        cnt, i, s = 0, 0, list(s)
        while i < len(s):
            size = min(k, len(s) - i)
            if cnt % 2 == 0:
                for j in range(size / 2): s[i + j], s[i + size - j - 1] = s[i + size - j - 1], s[i + j]
            i += size
            cnt += 1
        return ''.join(s)


# ============================================================================

# 542. 01 Matrix
# Difficulty: Medium
# link: https://leetcode.com/problems/01-matrix/
# Companies: Uber,Amazon
# Categories: Depth-first Search,Breadth-first Search

# ----------------------------------------------------------------------------

class Solution(object):
    def updateMatrix(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: List[List[int]]
        """
        def get_adj(i, j):
            return filter(lambda pos: 0 <= pos[0] < len(matrix) and 0 <= pos[1] < len(matrix[0]),
                          [[i - 1, j], [i + 1, j], [i, j - 1], [i, j + 1]])
        bfs = []
        for i in xrange(len(matrix)):
            for j in xrange(len(matrix[0])):
                if matrix[i][j]:
                    set_none = True
                    for adj_i, adj_j in get_adj(i, j):
                        set_none &= matrix[adj_i][adj_j] == 1 or matrix[adj_i][adj_j] is None
                    if set_none:
                        matrix[i][j] = None
                    else:
                        bfs.append([i, j])
        bfs2 = []
        level = 1
        while bfs:
            level += 1
            while bfs:
                i, j = bfs.pop()
                for adj_i, adj_j in get_adj(i, j):
                    if matrix[adj_i][adj_j] is None:
                        matrix[adj_i][adj_j] = level
                        bfs2.append([adj_i, adj_j])
            bfs, bfs2 = bfs2, bfs
        return matrix


# ============================================================================

# 543. Diameter of Binary Tree
# Difficulty: Easy
# link: https://leetcode.com/problems/diameter-of-binary-tree/
# Companies: Qualtrics,Atlassian,Amazon,Facebook,Bloomberg,Microsoft
# Categories: Tree

# ----------------------------------------------------------------------------

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def diameterOfBinaryTree(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        self.max = 0
        def _longest_len(node):
            if not node: return 0
            l, r = _longest_len(node.left), _longest_len(node.right)
            self.max = max(self.max, l + r)
            return max(l, r) + 1
        _longest_len(root)
        return self.max


# ============================================================================

# 547. Friend Circles
# Difficulty: Medium
# link: https://leetcode.com/problems/friend-circles/
# Companies: Uber,Twitter,Pocket Gems,Two Sigma,Amazon
# Categories: Depth-first Search,Union Find

# ----------------------------------------------------------------------------

class Solution(object):
    def findCircleNum(self, M):
        uf = list(range(len(M)))
        for i, node in enumerate(M):
            for j, is_adj in enumerate(node):
                if is_adj:
                    path = []
                    for node in [i, j]:
                        path.append(node)
                        while uf[path[-1]] != path[-1]: path.append(uf[path[-1]])
                    for node in path: uf[node] = path[-1]

        return sum(node == comp for node, comp in enumerate(uf))


# ============================================================================

# 551. Student Attendance Record I
# Difficulty: Easy
# link: https://leetcode.com/problems/student-attendance-record-i/
# Companies: Google
# Categories: String

# ----------------------------------------------------------------------------

class Solution(object):
    def checkRecord(self, s):
        """
        :type s: str
        :rtype: bool
        """
        return (s.count('A') <= 1) and ('LLL' not in s)


# ============================================================================

# 553. Optimal Division
# Difficulty: Medium
# link: https://leetcode.com/problems/optimal-division/
# Companies: Amazon
# Categories: Math,String

# ----------------------------------------------------------------------------

class Solution(object):
    def optimalDivision(self, nums):
        """
        :type nums: List[int]
        :rtype: str
        """
        return '/'.join(map(str, nums)) if len(nums) <= 2 else \
            '%d/(%s)' %(nums[0], '/'.join(map(str, nums[1:])))


# ============================================================================

# 554. Brick Wall
# Difficulty: Medium
# link: https://leetcode.com/problems/brick-wall/
# Companies: Oracle
# Categories: Hash Table

# ----------------------------------------------------------------------------

class Solution(object):
    def leastBricks(self, wall):
        """
        :type wall: List[List[int]]
        :rtype: int
        """
        split_count = {}
        split_max = 0
        split_idx = 0
        for row in wall:
            cur_sum = 0
            for i in range(0, len(row) - 1):
                cur_sum += row[i]
                split_count.setdefault(cur_sum, 0)
                split_count[cur_sum] += 1
                split_max = max(split_count[cur_sum], split_max)
        return len(wall) - split_max


# ============================================================================

# 556. Next Greater Element III
# Difficulty: Medium
# link: https://leetcode.com/problems/next-greater-element-iii/
# Companies: Amazon
# Categories: String

# ----------------------------------------------------------------------------

class Solution(object):
    def nextGreaterElement(self, n):
        """
        :type n: int
        :rtype: int
        """
        num = list(str(n))
        n = len(num)
        for i in xrange(n - 2, -1, -1):
            if num[i] < num[i + 1]:
                next_largest = i + 1
                for j in xrange(i + 2, n):
                    if num[i] < num[j] <= num[next_largest]:
                        next_largest = j
                num[i], num[next_largest] = num[next_largest], num[i]
                next_greater = int(''.join(num[:i + 1] + num[i + 1:][::-1]))
                return next_greater if next_greater < (1<<31) else -1
        return -1


# ============================================================================

# 557. Reverse Words in a String III
# Difficulty: Easy
# link: https://leetcode.com/problems/reverse-words-in-a-string-iii/
# Companies: Microsoft,Snapchat,Yahoo
# Categories: String

# ----------------------------------------------------------------------------

class Solution(object):
    def reverseWords(self, s): return ' '.join([word[::-1] for word in s.split(' ')])


# ============================================================================

# 560. Subarray Sum Equals K
# Difficulty: Medium
# link: https://leetcode.com/problems/subarray-sum-equals-k/
# Companies: Google,Goldman Sachs,Amazon,Facebook,Oracle,eBay,Microsoft
# Categories: Array,Hash Table

# ----------------------------------------------------------------------------

class Solution(object):
    def subarraySum(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        sum_from_left = 0
        sum_count = {0:1}
        res = 0
        for i, num in enumerate(nums):
            sum_from_left += num
            target_key = sum_from_left - k
            if target_key in sum_count:
                res += sum_count[target_key]
            sum_count.setdefault(sum_from_left, 0)
            sum_count[sum_from_left] += 1
        return res


# ============================================================================

# 561. Array Partition I
# Difficulty: Easy
# link: https://leetcode.com/problems/array-partition-i/
# Companies: Amazon,Apple
# Categories: Array

# ----------------------------------------------------------------------------

class Solution(object):
    def arrayPairSum(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        nums.sort()
        return sum(nums[::2])


# ============================================================================

# 563. Binary Tree Tilt
# Difficulty: Easy
# link: https://leetcode.com/problems/binary-tree-tilt/
# Companies: Indeed
# Categories: Tree

# ----------------------------------------------------------------------------

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def findTilt(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        self.tilt = 0
        def get_sum_update_tilt(node):
            if not node: return 0
            l, r = get_sum_update_tilt(node.left), get_sum_update_tilt(node.right)
            self.tilt += abs(l - r)
            return l + r + node.val
        get_sum_update_tilt(root)
        return self.tilt


# ============================================================================

# 565. Array Nesting
# Difficulty: Medium
# link: https://leetcode.com/problems/array-nesting/
# Companies: Google,Apple
# Categories: Array

# ----------------------------------------------------------------------------

class Solution(object):
    def arrayNesting(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        max_dep = 0
        for i in range(len(nums)):
            cur, depth = i, 0
            while nums[cur] is not None:
                nums[cur], cur = None, nums[cur]
                depth += 1
            max_dep = max(depth, max_dep)
        return max_dep


# ============================================================================

# 566. Reshape the Matrix
# Difficulty: Easy
# link: https://leetcode.com/problems/reshape-the-matrix/
# Companies: Mathworks
# Categories: Array

# ----------------------------------------------------------------------------

class Solution(object):
    def matrixReshape(self, nums, r, c):
        """
        :type nums: List[List[int]]
        :type r: int
        :type c: int
        :rtype: List[List[int]]
        """
        if r * c != len(nums) * len(nums[0]): return nums
        res = [num for row in nums for num in row]
        return [res[i:i+c] for i in range(0, len(res), c)]


# ============================================================================

# 567. Permutation in String
# Difficulty: Medium
# link: https://leetcode.com/problems/permutation-in-string/
# Companies: Uber,Amazon,Facebook
# Categories: Two Pointers,Sliding Window

# ----------------------------------------------------------------------------

class Solution(object):
    def checkInclusion(self, s1, s2):
        """
        :type s1: str
        :type s2: str
        :rtype: bool
        """
        from collections import Counter
        chr_cnt = Counter(s1)
        for right in range(len(s2)):
            r = s2[right]
            chr_cnt[r] = chr_cnt.get(r, 0) - 1
            if not chr_cnt[r]: del chr_cnt[r]
            if right >= len(s1):
                l = s2[right - len(s1)]
                chr_cnt[l] = chr_cnt.get(l, 0) + 1
                if not chr_cnt[l]: del chr_cnt[l]
            if len(chr_cnt) == 0: return True
        return False


# ============================================================================

# 572. Subtree of Another Tree
# Difficulty: Easy
# link: https://leetcode.com/problems/subtree-of-another-tree/
# Companies: Amazon,Google,Microsoft
# Categories: Tree

# ----------------------------------------------------------------------------

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def isSubtree(self, s, t):
        """
        :type s: TreeNode
        :type t: TreeNode
        :rtype: bool
        """
        def serialize(node):
            if not node: return ''
            return '[%d,%s,%s]' %(node.val, serialize(node.left), serialize(node.right))
        return serialize(t) in serialize(s)


# ============================================================================

# 575. Distribute Candies
# Difficulty: Easy
# link: https://leetcode.com/problems/distribute-candies/
# Companies: Microsoft
# Categories: Hash Table

# ----------------------------------------------------------------------------

class Solution(object):
    def distributeCandies(self, candies):
        """
        :type candies: List[int]
        :rtype: int
        """
        return min(len(candies)/2, len(set(candies)))


# ============================================================================

# 576. Out of Boundary Paths
# Difficulty: Medium
# link: https://leetcode.com/problems/out-of-boundary-paths/
# Companies: Baidu
# Categories: Dynamic Programming,Depth-first Search

# ----------------------------------------------------------------------------

class Solution(object):
    def findPaths(self, m, n, N, i, j):
        """
        :type m: int
        :type n: int
        :type N: int
        :type i: int
        :type j: int
        :rtype: int
        """
        grid = [[0] * n for _ in range(m)]
        dif = zip([0, 1, 0, -1],
                  [1, 0, -1, 0])
        for _ in range(N):
            grid = [[
                sum(
                    grid[adj_x][adj_y] if (0 <= adj_x < m and 0 <= adj_y < n) else 1
                    for adj_x, adj_y in [(x + x_dir, y + y_dir) for x_dir, y_dir in dif])
                for y in range(n)]
                for x in range(m)
            ]
        return grid[i][j] % (10**9 + 7)


# ============================================================================

# 581. Shortest Unsorted Continuous Subarray
# Difficulty: Easy
# link: https://leetcode.com/problems/shortest-unsorted-continuous-subarray/
# Companies: Amazon
# Categories: Array

# ----------------------------------------------------------------------------

class Solution(object):
    def findUnsortedSubarray(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        sorted_nums = sorted(nums)
        i, j = 0, len(nums) - 1
        while i < j:
            if nums[i] == sorted_nums[i]: i += 1
            elif nums[j] == sorted_nums[j]: j -= 1
            else: return j - i + 1
        return 0


# ============================================================================

# 583. Delete Operation for Two Strings
# Difficulty: Medium
# link: https://leetcode.com/problems/delete-operation-for-two-strings/
# Companies: Google
# Categories: String

# ----------------------------------------------------------------------------

class Solution(object):
    def minDistance(self, word1, word2):
        """
        :type word1: str
        :type word2: str
        :rtype: int
        """
        if not word1 or not word2: return len(word1) or len(word2)
        dp = [[0] * (len(word2) + 1) for _ in range(len(word1) + 1)]
        for i in range(len(word1)):
            for j in range(len(word2)):
                dp[i + 1][j + 1] = max(dp[i][j + 1], dp[i + 1][j], dp[i][j] + (1 if word1[i] == word2[j] else 0))
        return (len(word1) + len(word2) - dp[-1][-1] * 2)


# ============================================================================

# 593. Valid Square
# Difficulty: Medium
# link: https://leetcode.com/problems/valid-square/
# Companies: Pure Storage
# Categories: Math

# ----------------------------------------------------------------------------

class Solution(object):
    def validSquare(self, p1, p2, p3, p4):
        """
        :type p1: List[int]
        :type p2: List[int]
        :type p3: List[int]
        :type p4: List[int]
        :rtype: bool
        """
        points = [p1, p2, p3, p4]
        dists = [ ((points[i][0] - points[j][0])**2 + (points[i][1] - points[j][1])**2)
                 for i in range(len(points)) for j in range(i + 1, len(points))]
        from collections import Counter
        dists = Counter(dists)
        keys = dists.keys()
        return len(keys) == 2 and \
            (dists[keys[0]] == 2 or dists[keys[0]] == 4) and \
            (dists[keys[1]] == 2 or dists[keys[1]] == 4)


# ============================================================================

# 594. Longest Harmonious Subsequence
# Difficulty: Easy
# link: https://leetcode.com/problems/longest-harmonious-subsequence/
# Companies: LiveRamp
# Categories: Hash Table

# ----------------------------------------------------------------------------

class Solution(object):
    def findLHS(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        from collections import Counter
        counts = Counter(nums)
        return max([counts[x] + counts[x + 1] for x in counts if x + 1 in counts] or [0])


# ============================================================================

# 598. Range Addition II
# Difficulty: Easy
# link: https://leetcode.com/problems/range-addition-ii/
# Companies: IXL
# Categories: Math

# ----------------------------------------------------------------------------

class Solution(object):
    def maxCount(self, m, n, ops):
        """
        :type m: int
        :type n: int
        :type ops: List[List[int]]
        :rtype: int
        """
        max_x = min([op[0] for op in ops if op[0]] or [0])
        max_y = min([op[1] for op in ops if op[1]] or [0])
        return (max_x * max_y) or (m * n)


# ============================================================================

# 599. Minimum Index Sum of Two Lists
# Difficulty: Easy
# link: https://leetcode.com/problems/minimum-index-sum-of-two-lists/
# Companies: Oracle,Yelp
# Categories: Hash Table

# ----------------------------------------------------------------------------

class Solution(object):
    def findRestaurant(self, list1, list2):
        """
        :type list1: List[str]
        :type list2: List[str]
        :rtype: List[str]
        """
        rest_to_idx_2 = {rest: i for i, rest in enumerate(list2)}
        min_dist, min_rests = float('inf'), []
        for i, rest in enumerate(list1):
            if rest in rest_to_idx_2:
                fav_sum = rest_to_idx_2[rest] + i
                if fav_sum < min_dist: min_dist, min_rests = fav_sum, [rest]
                elif fav_sum == min_dist: min_rests.append(rest)

        return min_rests


# ============================================================================

# 605. Can Place Flowers
# Difficulty: Easy
# link: https://leetcode.com/problems/can-place-flowers/
# Companies: LinkedIn
# Categories: Array

# ----------------------------------------------------------------------------

class Solution(object):
    def canPlaceFlowers(self, flowerbed, n):
        """
        :type flowerbed: List[int]
        :type n: int
        :rtype: bool
        """
        count = 0
        for i, planted in enumerate(flowerbed):
            if not planted:
                if (i == 0 or not flowerbed[i-1]) and (i == (len(flowerbed) - 1) or not flowerbed[i+1]):
                    flowerbed[i] = 1
                    count += 1
            if count >= n: return True
        return False


# ============================================================================

# 606. Construct String from Binary Tree
# Difficulty: Easy
# link: https://leetcode.com/problems/construct-string-from-binary-tree/
# Companies: Amazon
# Categories: String,Tree

# ----------------------------------------------------------------------------

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def tree2str(self, t):
        """
        :type t: TreeNode
        :rtype: str
        """
        def _tree2str(node):
            if not node: return ''
            left = _tree2str(node.left)
            right = _tree2str(node.right)
            if not left and not right: return str(node.val)
            elif not right: return '%d(%s)' %(node.val, left)
            else: return '%d(%s)(%s)' %(node.val, left, right)
        return _tree2str(t)


# ============================================================================

# 609. Find Duplicate File in System
# Difficulty: Medium
# link: https://leetcode.com/problems/find-duplicate-file-in-system/
# Companies: Dropbox,Amazon
# Categories: Hash Table,String

# ----------------------------------------------------------------------------

class Solution(object):
    def findDuplicate(self, paths):
        """
        :type paths: List[str]
        :rtype: List[List[str]]
        """
        from collections import defaultdict
        content_to_path = defaultdict(list)
        for p_f_c in paths:
            p_f_c = p_f_c.split(' ')
            folder = p_f_c[0]
            for i in range(1, len(p_f_c)):
                file_name, content = p_f_c[i][:-1].split('(')
                content_to_path[content].append('%s/%s' %(folder, file_name))
        return [paths for paths in content_to_path.values() if len(paths) > 1]


# ============================================================================

# 611. Valid Triangle Number
# Difficulty: Medium
# link: https://leetcode.com/problems/valid-triangle-number/
# Companies: Bloomberg,LinkedIn
# Categories: Array

# ----------------------------------------------------------------------------

class Solution(object):
    def triangleNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        res = 0
        nums.sort()
        res = 0
        for i in range(len(nums) -1, 1, -1):
            l, r = 0, i - 1
            while l < r:
                if nums[l] + nums[r] > nums[i]:
                    res += (r - l)
                    r -= 1
                else:
                    l += 1
        return res


# ============================================================================

# 617. Merge Two Binary Trees
# Difficulty: Easy
# link: https://leetcode.com/problems/merge-two-binary-trees/
# Companies: Amazon
# Categories: Tree

# ----------------------------------------------------------------------------

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def mergeTrees(self, t1, t2):
        """
        :type t1: TreeNode
        :type t2: TreeNode
        :rtype: TreeNode
        """
        def _merge_nodes(node1, node2):
            if not (node1 and node2): return node1 or node2
            node1.val += node2.val
            node1.left = _merge_nodes(node1.left, node2.left)
            node1.right = _merge_nodes(node1.right, node2.right)
            return node1
        return _merge_nodes(t1, t2)


# ============================================================================

# 623. Add One Row to Tree
# Difficulty: Medium
# link: https://leetcode.com/problems/add-one-row-to-tree/
# Companies: Gilt Groupe,Microsoft
# Categories: Tree

# ----------------------------------------------------------------------------

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def addOneRow(self, root, v, d):
        """
        :type root: TreeNode
        :type v: int
        :type d: int
        :rtype: TreeNode
        """
        if d == 1:
            new_node = TreeNode(v)
            new_node.left = root
            return new_node
        bfs = [root]
        prev = None
        for i in range(d - 2):
            bfs = [kid for node in bfs for kid in [node.left, node.right] if kid]
        for node in bfs:
            new_node = TreeNode(v)
            node.left, new_node.left = new_node, node.left
            new_node = TreeNode(v)
            node.right, new_node.right = new_node, node.right
        return root


# ============================================================================

# 628. Maximum Product of Three Numbers
# Difficulty: Easy
# link: https://leetcode.com/problems/maximum-product-of-three-numbers/
# Companies: Amazon
# Categories: Array,Math

# ----------------------------------------------------------------------------

class Solution(object):
    def maximumProduct(self, nums):
        max3 = []
        min2 = []
        for num in nums:
            if not max3.append(num) and len(max3) == 4: max3.remove(min(max3))
            if not min2.append(num) and len(min2) == 3: min2.remove(max(min2))
        def prod(lst): return reduce(lambda x, y: x * y, lst, 1)
        return max(prod(max3), prod(min2 + [max(max3)]))


# ============================================================================

# 630. Course Schedule III
# Difficulty: Hard
# link: https://leetcode.com/problems/course-schedule-iii/
# Companies: Google
# Categories: Greedy

# ----------------------------------------------------------------------------

class Solution(object):
    def scheduleCourse(self, courses):
        """
        :type courses: List[List[int]]
        :rtype: int
        """
        import heapq
        courses = sorted(map(lambda x: list(reversed(x)), courses))
        max_heap = []
        max_len = total_dur = 0
        for deadline, dur in courses:
            total_dur += dur
            heapq.heappush(max_heap, -dur)
            while total_dur > deadline:
                total_dur += heapq.heappop(max_heap)
            max_len = max(max_len, len(max_heap))
        return max_len


# ============================================================================

# 633. Sum of Square Numbers
# Difficulty: Easy
# link: https://leetcode.com/problems/sum-of-square-numbers/
# Companies: Facebook,LinkedIn
# Categories: Math

# ----------------------------------------------------------------------------

class Solution(object):
    def judgeSquareSum(self, c):
        a, b = 0, int(c ** (0.5))
        dp = [0]
        inc = 1
        for i in range(b + 1):
            dp.append(inc + dp[-1])
            inc += 2
        while a <= b:
            eq = dp[a] + dp[b]
            if eq == c: return True
            elif eq < c: a += 1
            elif eq > c: b -= 1
        return False


# ============================================================================

# 637. Average of Levels in Binary Tree
# Difficulty: Easy
# link: https://leetcode.com/problems/average-of-levels-in-binary-tree/
# Companies: Facebook
# Categories: Tree

# ----------------------------------------------------------------------------

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def averageOfLevels(self, root):
        """
        :type root: TreeNode
        :rtype: List[float]
        """
        if not root: return []
        avgs = []
        bfs = [root]
        while bfs:
            vals = [node.val for node in bfs]
            avgs.append(float(sum(vals)) / len(vals))
            bfs = [kid for node in bfs for kid in [node.left, node.right] if kid]
        return avgs


# ============================================================================

# 638. Shopping Offers
# Difficulty: Medium
# link: https://leetcode.com/problems/shopping-offers/
# Companies: Google
# Categories: Dynamic Programming,Depth-first Search

# ----------------------------------------------------------------------------

class Solution(object):
    def shoppingOffers(self, price, special, needs):
        def dfs(total=0, idx=0):
            if idx == len(special):
                return total + sum(need * price[i] for i, need in enumerate(needs))
            cost = []
            n, quan = len(needs), 0
            while all(needs[i] - quan * special[idx][i] >= 0 for i in range(n)):
                for i in range(n): needs[i] = needs[i] - quan * special[idx][i]
                cost.append(dfs(total + special[idx][-1] * quan, idx + 1))
                for i in range(n): needs[i] = needs[i] + quan * special[idx][i]
                quan += 1
            return min(c for c in cost)
        return dfs()


# ============================================================================

# 640. Solve the Equation
# Difficulty: Medium
# link: https://leetcode.com/problems/solve-the-equation/
# Companies: Uber,Amazon,Google
# Categories: Math

# ----------------------------------------------------------------------------

class Solution(object):
    def solveEquation(self, equation):
        """
        :type equation: str
        :rtype: str
        """
        import re
        def count_x(s):
            return sum(int(num_x)
                       for num_x in re.findall('[\+-]\d*(?=x)',
                                               re.sub('(?<=[\+-])(?=x)', '1', s)))
        def count_num(s): return sum(int(num) for num in re.findall('[\+-]\d+(?=[+\-])', s + '+'))
        left, right = [(x if x.startswith('-') else '+' + x) for x in equation.split('=')]
        num_x = count_x(left) - count_x(right)
        val = count_num(right) - count_num(left)
        if not num_x and not val: return "Infinite solutions"
        elif not num_x: return 'No solution'
        return 'x=%d' % (val / num_x)


# ============================================================================

# 643. Maximum Average Subarray I
# Difficulty: Easy
# link: https://leetcode.com/problems/maximum-average-subarray-i/
# Companies: Google
# Categories: Array

# ----------------------------------------------------------------------------

class Solution(object):
    def findMaxAverage(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: float
        """
        for i in range(len(nums) - 2, -1, -1): nums[i] += nums[i + 1]
        nums.append(0)
        max_avg = float('-inf')
        for i in range(len(nums) - k):max_avg = max(max_avg, (float(nums[i]) - nums[i + k]) / k)
        return max_avg


# ============================================================================

# 645. Set Mismatch
# Difficulty: Easy
# link: https://leetcode.com/problems/set-mismatch/
# Companies: Amazon
# Categories: Hash Table,Math

# ----------------------------------------------------------------------------

class Solution(object):
    def findErrorNums(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        for num in nums:
            if nums[abs(num) - 1] < 0:
                dup = abs(num)
                break
            nums[abs(num) - 1] = -nums[abs(num) - 1]
        for i in range(len(nums)): nums[i] = abs(nums[i])
        n = len(nums)
        return dup, ((n * (n + 1) / 2)) - sum(nums) + dup


# ============================================================================

# 646. Maximum Length of Pair Chain
# Difficulty: Medium
# link: https://leetcode.com/problems/maximum-length-of-pair-chain/
# Companies: Bloomberg
# Categories: Dynamic Programming

# ----------------------------------------------------------------------------

class Solution(object):
    def findLongestChain(self, pairs):
        """
        :type pairs: List[List[int]]
        :rtype: int
        """
        pairs.sort()
        cur = None
        count = 0
        for itv in reversed(pairs):
            if cur == None or cur[0] > itv[1]:
                cur = itv
                count += 1
        return count


# ============================================================================

# 647. Palindromic Substrings
# Difficulty: Medium
# link: https://leetcode.com/problems/palindromic-substrings/
# Companies: Twitter,Facebook,Amazon,Citadel
# Categories: String,Dynamic Programming

# ----------------------------------------------------------------------------

class Solution(object):
    def countSubstrings(self, s):
        """
        :type s: str
        :rtype: int
        """
        return sum(s[i:j] == s[i:j][::-1] for j in range(len(s) + 1) for i in range(j))


# ============================================================================

# 648. Replace Words
# Difficulty: Medium
# link: https://leetcode.com/problems/replace-words/
# Companies: Uber
# Categories: Hash Table,Trie

# ----------------------------------------------------------------------------

class Solution(object):
    def replaceWords(self, dict, sentence):
        """
        :type dict: List[str]
        :type sentence: str
        :rtype: str
        """
        hashed = set()
        for word in dict:
            hashed.add(hash(word))
        res = []
        for word in sentence.split(' '):
            replaced_word = word
            for i in range(1, len(word)):
                if hash(word[:i]) in hashed:
                    replaced_word = word[:i]
                    break
            res.append(replaced_word)
        return ' '.join(res)


# ============================================================================

# 652. Find Duplicate Subtrees
# Difficulty: Medium
# link: https://leetcode.com/problems/find-duplicate-subtrees/
# Companies: Bloomberg
# Categories: Tree

# ----------------------------------------------------------------------------

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def findDuplicateSubtrees(self, root):
        """
        :type root: TreeNode
        :rtype: List[TreeNode]
        """
        def _findDuplicateSubtrees(node, hash_to_count_node):
            if not node: return
            l_res = _findDuplicateSubtrees(node.left, hash_to_count_node)
            r_res = _findDuplicateSubtrees(node.right, hash_to_count_node)
            serial = (node.val, l_res, r_res)
            hash_val = hash(str(serial))
            hash_to_count_node.setdefault(hash_val, [0, node])
            hash_to_count_node[hash_val][0] += 1
            return serial

        hash_to_count_node = {}
        _findDuplicateSubtrees(root, hash_to_count_node)
        return [node for i, node in hash_to_count_node.values() if i >= 2]


# ============================================================================

# 653. Two Sum IV - Input is a BST
# Difficulty: Easy
# link: https://leetcode.com/problems/two-sum-iv-input-is-a-bst/
# Companies: Amazon
# Categories: Tree

# ----------------------------------------------------------------------------

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def findTarget(self, root, k):
        if not root: return False
        visited = set()
        bfs = [root]
        while bfs:
            for node in bfs:
                if (k - node.val) in visited: return True
                visited.add(node.val)
            bfs = [kid for node in bfs for kid in [node.left, node.right] if kid]
        return False


# ============================================================================

# 654. Maximum Binary Tree
# Difficulty: Medium
# link: https://leetcode.com/problems/maximum-binary-tree/
# Companies: Amazon
# Categories: Tree

# ----------------------------------------------------------------------------

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def constructMaximumBinaryTree(self, nums):
        """
        :type nums: List[int]
        :rtype: TreeNode
        """
        stack = []
        for num in nums:
            node = TreeNode(num)
            while stack and stack[-1].val < num:
                node.left = stack.pop()
            if stack:
                stack[-1].right = node
            stack.append(node)
        return stack[0]


# ============================================================================

# 655. Print Binary Tree
# Difficulty: Medium
# link: https://leetcode.com/problems/print-binary-tree/
# Companies: Uber,LinkedIn
# Categories: Tree

# ----------------------------------------------------------------------------

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def printTree(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[str]]
        """
        bfs = [root]
        depth = 0
        while bfs:
            depth += 1
            bfs = [child for node in bfs for child in [node.right, node.left] if child]

        # 1->1, 2->3, 3->7, 4->15, 5->31
        # 2^1-1=1, 2^2-1=3, 2^3-1=7, 2^4-1=15, 2^1-1=1

        width = 2 ** depth - 1
        res = [['']*width for _ in range(depth)]

        def _build(node=root, l=0, r=width, lvl=0):
            if not node: return res
            mid = (l + r) / 2
            res[lvl][mid] = str(node.val)
            _build(node.left, l, mid, lvl + 1)
            _build(node.right, mid + 1, r, lvl + 1)
            return res

        return _build()


# ============================================================================

# 657. Robot Return to Origin
# Difficulty: Easy
# link: https://leetcode.com/problems/robot-return-to-origin/
# Companies: Goldman Sachs
# Categories: String

# ----------------------------------------------------------------------------

class Solution(object):
    def judgeCircle(self, moves):
        """
        :type moves: str
        :rtype: bool
        """
        from collections import Counter
        counts = Counter(moves)
        return counts.get('U', 0) == counts.get('D', 0) and counts.get('L', 0) == counts.get('R', 0)


# ============================================================================

# 658. Find K Closest Elements
# Difficulty: Medium
# link: https://leetcode.com/problems/find-k-closest-elements/
# Companies: Snapchat
# Categories: Binary Search

# ----------------------------------------------------------------------------

class Solution(object):
    def findClosestElements(self, arr, k, x):
        i, j = 0, len(arr)
        while i < j:
            mid = (i + j) / 2
            if x < arr[mid]: j = mid
            elif arr[mid] < x: i = mid + 1
            else: break
        i = j = mid
        size = 1
        while size < len(arr) and size < k :
            if (j == len(arr) - 1) or (i > 0) and (x - arr[i - 1]) <= (arr[j + 1] - x): i -= 1
            else: j += 1
            size += 1
        return arr[i: j + 1]


# ============================================================================

# 659. Split Array into Consecutive Subsequences
# Difficulty: Medium
# link: https://leetcode.com/problems/split-array-into-consecutive-subsequences/
# Companies: Google
# Categories: Heap,Greedy

# ----------------------------------------------------------------------------

class Solution(object):
    def isPossible(self, nums):
        from collections import Counter
        cnt = Counter(nums)
        tail = Counter()
        for n in nums:
            if cnt[n]:
                cnt[n] -= 1
                if tail[n]:
                    tail[n] -= 1
                    tail[n+1] += 1
                elif cnt[n+1] and cnt[n+2]:
                    cnt[n+1]-=1
                    cnt[n+2]-=1
                    tail[n+3]+= 1
                else:
                    return False
        return True


# ============================================================================

# 661. Image Smoother
# Difficulty: Easy
# link: https://leetcode.com/problems/image-smoother/
# Companies: Apple
# Categories: Array

# ----------------------------------------------------------------------------

class Solution(object):
    def imageSmoother(self, M):
        if not M or not M[0]: return M
        m, n = len(M), len(M[0])
        res = [[None] * n for _ in range(m)]
        for i in range(m):
            for j in range(n):
                pos = [(i + difx, j + dify) for difx in [-1, 0, 1] for dify in [-1, 0, 1]]
                adjs = [M[x][y] for x, y in pos if 0 <= x < m and 0 <= y < n]
                res[i][j] = sum(adjs) / len(adjs)
        return res


# ============================================================================

# 662. Maximum Width of Binary Tree
# Difficulty: Medium
# link: https://leetcode.com/problems/maximum-width-of-binary-tree/
# Companies: Bloomberg
# Categories: Tree

# ----------------------------------------------------------------------------

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def widthOfBinaryTree(self, root):
        if not root: return
        bfs = [(root, 1)]
        max_len = 0
        while bfs:
            max_len = max(max_len, bfs[-1][1] - bfs[0][1] + 1)
            bfs = [(kid, pos * 2 + (kid == node.right))
                   for node, pos in bfs
                   for kid in (node.left, node.right) if kid]
        return max_len


# ============================================================================

# 665. Non-decreasing Array
# Difficulty: Easy
# link: https://leetcode.com/problems/non-decreasing-array/
# Companies: Google
# Categories: Array

# ----------------------------------------------------------------------------

class Solution(object):
    def checkPossibility(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        modified = False
        for i in range(1, len(nums)):
            if nums[i - 1] > nums[i]:
                if modified: return False
                if i - 2 < 0 or nums[i - 2] <= nums[i]: nums[i - 1] = nums[i]
                else: nums[i] = nums[i - 1]
                modified = True
        return True


# ============================================================================

# 667. Beautiful Arrangement II
# Difficulty: Medium
# link: https://leetcode.com/problems/beautiful-arrangement-ii/
# Companies: Google
# Categories: Array

# ----------------------------------------------------------------------------

class Solution(object):
    def constructArray(self, n, k):
        """
        :type n: int
        :type k: int
        :rtype: List[int]
        """
        if k == 1: return range(1, n + 1)
        res = []
        i, j = 1, k + 1
        while i <= j:
            res.extend([j, i])
            i += 1
            j -= 1
        if len(res) != k + 1: res.pop()
        res.extend(range(k + 2, n + 1))
        return res


# ============================================================================

# 669. Trim a Binary Search Tree
# Difficulty: Easy
# link: https://leetcode.com/problems/trim-a-binary-search-tree/
# Companies: Adobe
# Categories: Tree

# ----------------------------------------------------------------------------

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def trimBST(self, root, L, R):
        """
        :type root: TreeNode
        :type L: int
        :type R: int
        :rtype: TreeNode
        """
        def _trimBST(node, L, R):
            if not node: return
            elif node.val < L: return _trimBST(node.right, L, R)
            elif node.val > R: return _trimBST(node.left, L, R)
            else:
                node.left = _trimBST(node.left, L, R)
                node.right = _trimBST(node.right, L, R)
                return node
        return _trimBST(root, L, R)


# ============================================================================

# 670. Maximum Swap
# Difficulty: Medium
# link: https://leetcode.com/problems/maximum-swap/
# Companies: Facebook,Microsoft,ByteDance
# Categories: Array,Math

# ----------------------------------------------------------------------------

class Solution(object):
    def maximumSwap(self, num):
        """
        :type num: int
        :rtype: int
        """
        num = [int(i) for i in list(str(num))]
        max_from_right = [None] * len(num)
        for i in range(len(num) - 1, -1, -1):
            if i == len(num) - 1 or num[max_from_right[i + 1]] < num[i]:
                max_from_right[i] = i
            else: max_from_right[i] = max_from_right[i + 1]
        for i in range(len(num)):
            if max_from_right[i] != i and num[i] != num[max_from_right[i]] :
                num[i], num[max_from_right[i]] = num[max_from_right[i]], num[i]
                break
        return int(''.join([str(item) for item in num]))


# ============================================================================

# 671. Second Minimum Node In a Binary Tree
# Difficulty: Easy
# link: https://leetcode.com/problems/second-minimum-node-in-a-binary-tree/
# Companies: Uber,LinkedIn
# Categories: Tree

# ----------------------------------------------------------------------------

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def findSecondMinimumValue(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        def next_mins(node, root_val, res=[]):
            if not node: return res
            if node.val != root_val: res.append(node.val)
            else:
                next_mins(node.left, root_val, res)
                next_mins(node.right, root_val, res)
                return res
        if not root: return -1
        next_mins = next_mins(root, root.val)
        return min(next_mins) if next_mins else -1


# ============================================================================

# 674. Longest Continuous Increasing Subsequence
# Difficulty: Easy
# link: https://leetcode.com/problems/longest-continuous-increasing-subsequence/
# Companies: Facebook
# Categories: Array

# ----------------------------------------------------------------------------

class Solution(object):
    def findLengthOfLCIS(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        nums.append(float('-inf'))
        i = max_len = 0
        for j in range(1, len(nums)):
            if nums[j-1] >= nums[j]:
                max_len = max(max_len, j - i)
                i = j
        return max_len


# ============================================================================

# 676. Implement Magic Dictionary
# Difficulty: Medium
# link: https://leetcode.com/problems/implement-magic-dictionary/
# Companies: Microsoft
# Categories: Hash Table,Trie

# ----------------------------------------------------------------------------

class MagicDictionary(object):

    # trie based
    def ___init__(self):
        self.dict = {}

    def _buildDict(self, dict):
        for word in dict:
            cur = self.dict
            for char in word: cur = cur.setdefault(char, {})
            cur["#"] = True

    def _search(self, word):

        def _cmp(i, cur):
            for j in range(i, len(word)):
                char = word[j]
                if char not in cur: return False
                cur = cur[char]
            return '#' in cur

        cur = self.dict
        for i, char in enumerate(word):
            for dic_char in cur:
                if char != dic_char and dic_char != '#':
                    if _cmp(i + 1, cur[dic_char]): return True
            if char not in cur: return False
            cur = cur[char]
        return False

    # hash based
    def __init__(self):
        self.words = {}

    def buildDict(self, dict):
        for word in dict:
            for i in range(len(word)):
                self.words.setdefault(word[:i] + '_' + word[i + 1:], set()).add(word[i])

    def search(self, word):
        for i, char in enumerate(word):
            adj = self.words.get(word[:i] + '_' + word[i + 1:], None)
            if adj is not None and adj - set(char): return True
        return False


# ============================================================================

# 677. Map Sum Pairs
# Difficulty: Medium
# link: https://leetcode.com/problems/map-sum-pairs/
# Companies: Akuna Capital
# Categories: Trie

# ----------------------------------------------------------------------------

class MapSum(object):

    # trie DP
    def __init__(self):
        self.trie = {}
        self.cost = {}

    def insert(self, key, val):
        delta = val - self.cost.get(key, 0)
        self.cost[key] = val
        cur = self.trie
        for char in key:
            cur = cur.setdefault(char,{})
            cur['val'] = cur.setdefault('val', 0) + delta

    def sum(self, prefix):
        cur = self.trie
        for char in prefix:
            if char in cur: cur = cur[char]
            else: return 0
        return cur['val']

    # trie traverse
    def ___init__(self):
        self.trie = {}

    def _insert(self, key, val):
        cur = self.trie
        for char in key:
            cur = cur.setdefault(char, {})
        cur['val'] = val

    def _sum(self, prefix):
        cur = self.trie
        for char in prefix:
            if char in cur: cur = cur[char]
            else: return 0

        bfs = [cur]
        total = 0
        while bfs:
            bfs = [node[child] for node in bfs for child in node]
            total += sum((i for i in bfs if type(i) == int) or (0,))
            bfs = filter(lambda x: type(x) == dict, bfs)

        return total

# Your MapSum object will be instantiated and called as such:
# obj = MapSum()
# obj.insert(key,val)
# param_2 = obj.sum(prefix)


# ============================================================================

# 678. Valid Parenthesis String
# Difficulty: Medium
# link: https://leetcode.com/problems/valid-parenthesis-string/
# Companies: Amazon,Facebook
# Categories: String

# ----------------------------------------------------------------------------

class Solution(object):
    def checkValidString(self, s):
        """
        :type s: str
        :rtype: bool
        """
        high = low = 0
        for i, char in enumerate(s):
            high += -1 if char == ')' else 1
            low = low + 1 if char == '(' else max(low - 1, 0)
            if high < 0:
                return False
        return low == 0


# ============================================================================

# 680. Valid Palindrome II
# Difficulty: Easy
# link: https://leetcode.com/problems/valid-palindrome-ii/
# Companies: Facebook,Atlassian
# Categories: String

# ----------------------------------------------------------------------------

class Solution(object):
    def validPalindrome(self, s):
        """
        :type s: str
        :rtype: bool
        """
        for i in range(len(s) / 2):
            j = len(s) - i - 1
            if s[i] != s[j]:
                s1 = s[:i] + s[i + 1:]
                s2 = s[:j] + s[j + 1:]
                return s1 == s1[::-1] or s2 == s2[::-1]
        return True


# ============================================================================

# 682. Baseball Game
# Difficulty: Easy
# link: https://leetcode.com/problems/baseball-game/
# Companies: Amazon
# Categories: Stack

# ----------------------------------------------------------------------------

class Solution(object):
    def calPoints(self, ops):
        """
        :type ops: List[str]
        :rtype: int
        """
        stack = []
        for item in ops:
            if item == 'D': stack.append(stack[-1] * 2)
            elif item == '+': stack.append(stack[-1] + stack[-2])
            elif item == 'C': stack.pop()
            else: stack.append(int(item))
        return sum(stack)


# ============================================================================

# 684. Redundant Connection
# Difficulty: Medium
# link: https://leetcode.com/problems/redundant-connection/
# Companies: Amazon
# Categories: Tree,Union Find,Graph

# ----------------------------------------------------------------------------

class Solution(object):
    def findRedundantConnection(self, edges):
        """
        :type edges: List[List[int]]
        :rtype: List[int]
        """
        # use union find to find a connected component
        node_to_parent = range(len(edges) + 1)
        def get_root(node):
            path, cur = set(), node
            while cur != node_to_parent[cur]:
                path.add(cur)
                cur = node_to_parent[cur]
            root = cur
            for node in path: node_to_parent[node] = root
            return root
        for a, b in edges:
            root_a, root_b = get_root(a), get_root(b)
            if root_a == root_b: return [a, b]
            node_to_parent[root_a] = root_b


# ============================================================================

# 686. Repeated String Match
# Difficulty: Easy
# link: https://leetcode.com/problems/repeated-string-match/
# Companies: Google
# Categories: String

# ----------------------------------------------------------------------------

class Solution(object):
    def repeatedStringMatch(self, A, B):
        """
        :type A: str
        :type B: str
        :rtype: int
        """
        for i in range(3):
            times = len(B) / len(A) + i
            if B in A * times:
                return times
        return -1


# ============================================================================

# 687. Longest Univalue Path
# Difficulty: Easy
# link: https://leetcode.com/problems/longest-univalue-path/
# Companies: Amazon
# Categories: Tree,Recursion

# ----------------------------------------------------------------------------

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def longestUnivaluePath(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        self.max_len = 0
        def _longestUnivaluePath(node):
            if not node: return 0
            left_child = right_child = 0
            left_len, right_len = _longestUnivaluePath(node.left), _longestUnivaluePath(node.right)
            if node and node.left and node.left.val == node.val: left_child = left_len + 1
            if node and node.right and node.right.val == node.val: right_child = right_len + 1
            self.max_len = max(self.max_len, left_child + right_child)
            return max(left_child, right_child)
        _longestUnivaluePath(root)
        return self.max_len


# ============================================================================

# 688. Knight Probability in Chessboard
# Difficulty: Medium
# link: https://leetcode.com/problems/knight-probability-in-chessboard/
# Companies: Microsoft,Amazon,Apple,Goldman Sachs
# Categories: Dynamic Programming

# ----------------------------------------------------------------------------

class Solution(object):
    def knightProbability(self, N, K, r, c):
        """
        :type N: int
        :type K: int
        :type r: int
        :type c: int
        :rtype: float
        """
        x_y_diff = [2, -2, 1, -1]
        diff = [(x, y) for x in x_y_diff for y in x_y_diff if abs(x) != abs(y)]
        board = [[0] * N for _ in xrange(N)]
        board[r][c] = 1
        for _ in xrange(K):
            board = [[sum((board[x + x_dif][y + y_dif]
                           for x_dif, y_dif in diff
                           if 0 <= (x + x_dif) < N and 0 <= (y + y_dif) < N))
                      for y in xrange(N)]
                     for x in xrange(N)]
        return float(sum(map(sum, board))) / (len(diff) ** K)


# ============================================================================

# 690. Employee Importance
# Difficulty: Easy
# link: https://leetcode.com/problems/employee-importance/
# Companies: Uber,Amazon
# Categories: Hash Table,Depth-first Search,Breadth-first Search

# ----------------------------------------------------------------------------

"""
# Employee info
class Employee(object):
    def __init__(self, id, importance, subordinates):
        # It's the unique id of each node.
        # unique id of this employee
        self.id = id
        # the importance value of this employee
        self.importance = importance
        # the id of direct subordinates
        self.subordinates = subordinates
"""
class Solution(object):
    def getImportance(self, employees, id):
        """
        :type employees: Employee
        :type id: int
        :rtype: int
        """
        id_to_person = {person.id: person for person in employees}
        adjs = {person: [id_to_person[adj_id] for adj_id in person.subordinates] for person in employees}
        visited = set()
        bfs = [id_to_person[id]]
        importance = 0
        while bfs:
            importance += sum(person.importance for person in bfs)
            bfs = [adj for person in bfs for adj in adjs[person] if adj not in visited and (not visited.add(adj))]
        return importance


# ============================================================================

# 692. Top K Frequent Words
# Difficulty: Medium
# link: https://leetcode.com/problems/top-k-frequent-words/
# Companies: Expedia,TripAdvisor,Apple,Yelp,Amazon,Google,Bloomberg,Oracle,Microsoft
# Categories: Hash Table,Heap,Trie

# ----------------------------------------------------------------------------

class Solution(object):
    def topKFrequent(self, words, k):
        from collections import Counter
        import heapq
        freq_word = [(-cnt, word) for word, cnt in Counter(words).items()]
        heapq.heapify(freq_word)
        most_freq = float('inf')
        res = []
        while freq_word and (len(res) < k or -freq_word[0][0] == most_freq):
            most_freq, word = heapq.heappop(freq_word)
            most_freq *= 1
            res.append(word)
        return res


# ============================================================================

# 693. Binary Number with Alternating Bits
# Difficulty: Easy
# link: https://leetcode.com/problems/binary-number-with-alternating-bits/
# Companies: Yahoo
# Categories: Bit Manipulation

# ----------------------------------------------------------------------------

class Solution(object):
    def hasAlternatingBits(self, n):
        """
        :type n: int
        :rtype: bool
        """
        toggle = n & 1
        while n:
            n, is_one = divmod(n, 2)
            if toggle and not is_one or not toggle and is_one: return False
            toggle = not toggle
        return True


# ============================================================================

# 695. Max Area of Island
# Difficulty: Medium
# link: https://leetcode.com/problems/max-area-of-island/
# Companies: Qualtrics,Affirm,Wish,Amazon,Bloomberg,Twitch,DoorDash
# Categories: Array,Depth-first Search

# ----------------------------------------------------------------------------

class Solution(object):
    def maxAreaOfIsland(self, grid):

        if not len(grid) or not grid[0]: return 0
        m, n = len(grid), len(grid[0])

        max_size = 0
        for i in range(m):
            for j in range(n):
                if grid[i][j]:
                    size = 0
                    bfs = {(i, j)}
                    while bfs:
                        size += len(bfs)
                        for x, y in bfs: grid[x][y] = None
                        bfs = {(adj_x, adj_y) for x, y in bfs
                                              for adj_x, adj_y in [[x, y + 1], [x, y - 1], [x + 1, y], [x - 1, y]]
                                              if 0 <= adj_x < m and 0 <= adj_y < n and grid[adj_x][adj_y]}
                    max_size = max(max_size, size)

        for i in range(m):
            for j in range(n):
                grid[i][j] = 1 if grid[i][j] is None else grid[i][j]

        return max_size


# ============================================================================

# 696. Count Binary Substrings
# Difficulty: Easy
# link: https://leetcode.com/problems/count-binary-substrings/
# Companies: Helix
# Categories: String

# ----------------------------------------------------------------------------

class Solution(object):
    def countBinarySubstrings(self, s):
        """
        :type s: str
        :rtype: int
        """
        s = map(len, s.replace('01', '0 1').replace('10', '1 0').split())
        return sum([min(s[i], s[i + 1]) for i in range(len(s) - 1)])


# ============================================================================

# 697. Degree of an Array
# Difficulty: Easy
# link: https://leetcode.com/problems/degree-of-an-array/
# Companies: Mathworks,Robinhood,IXL,Citrix
# Categories: Array

# ----------------------------------------------------------------------------

class Solution(object):
    def findShortestSubArray(self, nums):
        from collections import Counter
        cnt = Counter(nums)
        max_freq = max(i for i in cnt.values())
        start_end = {}
        for i, num in enumerate(nums):
            if cnt[num] == max_freq:
                s, e = start_end.get(num, (i, i))
                start_end[num] = (min(i, s), max(i, e))
        return min(e - s + 1 for s, e in start_end.values())


# ============================================================================

# 698. Partition to K Equal Sum Subsets
# Difficulty: Medium
# link: https://leetcode.com/problems/partition-to-k-equal-sum-subsets/
# Companies: Amazon,LinkedIn,eBay
# Categories: Dynamic Programming,Recursion

# ----------------------------------------------------------------------------

class Solution(object):
    def canPartitionKSubsets(self, nums, k):
        from collections import Counter
        nums_sum = sum(nums)
        if nums_sum % k: return False
        per_bucket = nums_sum / k
        cnt = Counter(nums)
        nums.sort()
        def _canPartitionKSubsets(buckets=[per_bucket]*k, visited=set()):
            if not(any(buckets)): return True
            sorted_bucket = tuple(sorted(buckets))
            if sorted_bucket in visited: return False
            visited.add(sorted_bucket)
            num = nums.pop()
            for i in range(k):
                if buckets[i] - num >= 0:
                    buckets[i] -= num
                    if _canPartitionKSubsets(buckets): return True
                    buckets[i] += num
            nums.append(num)
            return False
        return _canPartitionKSubsets()


# ============================================================================

# 701. Insert into a Binary Search Tree
# Difficulty: Medium
# link: https://leetcode.com/problems/insert-into-a-binary-search-tree/
# Companies: Microsoft
# Categories: Tree

# ----------------------------------------------------------------------------

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def insertIntoBST(self, root, val):
        parent, cur = None, root
        while cur: parent, cur = cur, cur.right if cur.val < val else cur.left
        if val < parent.val: parent.left = TreeNode(val)
        else: parent.right = TreeNode(val)
        return root


# ============================================================================

# 713. Subarray Product Less Than K
# Difficulty: Medium
# link: https://leetcode.com/problems/subarray-product-less-than-k/
# Companies: Akuna Capital,Google
# Categories: Array,Two Pointers

# ----------------------------------------------------------------------------

class Solution(object):
    def numSubarrayProductLessThanK(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        if k == 0 : return 0
        prod = 1
        start = count = 0
        for end, elem in enumerate(nums):
            prod *= elem
            while prod >= k and start <= end:
                prod /= nums[start]
                start += 1
            count += end - start + 1
        return count


# ============================================================================

# 717. 1-bit and 2-bit Characters
# Difficulty: Easy
# link: https://leetcode.com/problems/1-bit-and-2-bit-characters/
# Companies: Quora,IXL
# Categories: Array

# ----------------------------------------------------------------------------

class Solution(object):
    def isOneBitCharacter(self, bits):
        """
        :type bits: List[int]
        :rtype: bool
        """
        return re.findall('(1.|0)', reduce(lambda x,y:x+str(y), bits, ''))[-1] == '0'


# ============================================================================

# 718. Maximum Length of Repeated Subarray
# Difficulty: Medium
# link: https://leetcode.com/problems/maximum-length-of-repeated-subarray/
# Companies: Indeed
# Categories: Array,Hash Table,Binary Search,Dynamic Programming

# ----------------------------------------------------------------------------

class Solution(object):
    def findLength(self, A, B):
        """
        :type A: List[int]
        :type B: List[int]
        :rtype: int
        """
        if not A or not B: return 0
        dp = [[0] * (len(B) + 1) for _ in xrange((len(A)) + 1)]
        for i in range(len(A)):
            for j in range(len(B)):
                if A[i] == B[j]: dp[i + 1][j + 1] = dp[i][j] + 1
        return max(item for row in dp for item in row)


# ============================================================================

# 720. Longest Word in Dictionary
# Difficulty: Easy
# link: https://leetcode.com/problems/longest-word-in-dictionary/
# Companies: Amazon
# Categories: Hash Table,Trie

# ----------------------------------------------------------------------------

class Solution(object):
    def longestWord(self, words):
        words_by_length = [set() for _ in range(max(map(len, words)) + 1)]
        for word in words: words_by_length[len(word)].add(word)
        for i in range(2, len(words_by_length)):
            for word in set(words_by_length[i]):
                if word[:-1] not in words_by_length[i - 1]: words_by_length[i].remove(word)
        return next((min(words) for words in reversed(words_by_length) if words))


# ============================================================================

# 721. Accounts Merge
# Difficulty: Medium
# link: https://leetcode.com/problems/accounts-merge/
# Companies: Houzz,Facebook,Microsoft
# Categories: Depth-first Search,Union Find

# ----------------------------------------------------------------------------

class Solution(object):
    def accountsMerge(self, accounts):

        # create union find memory
        uf = list(range(len(accounts)))
        def union(i, j):
            path = []
            for node in [i, j]:
                path.append(node)
                while path[-1] != uf[path[-1]]: path.append(uf[path[-1]])
            for i in path: uf[i] = path[-1]

        # union the connected nodes
        G = {}
        for i, acc in enumerate(accounts):
            for j in range(1, len(acc)):
                node = (acc[0], acc[j])
                if node not in G: G[node] = i
                else:
                    union(i, G[node])

        # reconstruct answer
        res = [[acc[0], set()] for acc in accounts]
        for i, acc in enumerate(accounts):
            root = i
            while uf[root] != root: root = uf[root]
            res[root][1] |= set(acc[1:])

        return [[node] + sorted(list(emails)) for i, (node, emails) in enumerate(res) if uf[i] == i]


# ============================================================================

# 724. Find Pivot Index
# Difficulty: Easy
# link: https://leetcode.com/problems/find-pivot-index/
# Companies: GoDaddy,Apple,Microsoft
# Categories: Array

# ----------------------------------------------------------------------------

class Solution(object):
    def pivotIndex(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        right_sums = nums + [0]
        for i in range(len(right_sums) - 2, -1, -1): right_sums[i] += right_sums[i + 1]
        left_sum = 0
        for i in range(len(nums)):
            if left_sum == right_sums[i + 1]: return i
            left_sum += nums[i]
        return -1


# ============================================================================

# 725. Split Linked List in Parts
# Difficulty: Medium
# link: https://leetcode.com/problems/split-linked-list-in-parts/
# Companies: Amazon,Google
# Categories: Linked List

# ----------------------------------------------------------------------------

# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def splitListToParts(self, root, k):
        cur, cnt = root, 0
        while cur: cur, cnt = cur.next, cnt + 1
        size, remainder = divmod(cnt, k)
        res = []
        cur = root
        for i in range(k):
            res.append(cur)
            prev = cur
            for j in range(size + bool(i < remainder)): prev, cur = cur, cur.next
            if prev: prev.next = None
        return res


# ============================================================================

# 728. Self Dividing Numbers
# Difficulty: Easy
# link: https://leetcode.com/problems/self-dividing-numbers/
# Companies: Adobe
# Categories: Math

# ----------------------------------------------------------------------------

class Solution(object):
    def selfDividingNumbers(self, left, right):
        """
        :type left: int
        :type right: int
        :rtype: List[int]
        """
        res = []
        for i in range(left, right + 1):
            x = i
            is_div = True
            while is_div and x:
                x, mod = divmod(x, 10)
                if not mod or i % mod != 0: is_div = False
            if is_div: res.append(i)
        return res


# ============================================================================

# 729. My Calendar I
# Difficulty: Medium
# link: https://leetcode.com/problems/my-calendar-i/
# Companies: Google
# Categories: Array

# ----------------------------------------------------------------------------

class MyCalendar(object):

    def __init__(self):
        self.intervals = []

    def book(self, start, end):
        """
        :type start: int
        :type end: int
        :rtype: bool
        """
        for int_start, int_end in self.intervals:
            if not (int_end <= start or end <= int_start):
                return False
        self.intervals.append((start,end))
        return True


# Your MyCalendar object will be instantiated and called as such:
# obj = MyCalendar()
# param_1 = obj.book(start,end)


# ============================================================================

# 733. Flood Fill
# Difficulty: Easy
# link: https://leetcode.com/problems/flood-fill/
# Companies: Amazon,Apple,Palantir Technologies,Microsoft
# Categories: Depth-first Search

# ----------------------------------------------------------------------------

class Solution(object):
    def floodFill(self, image, sr, sc, newColor):
        """
        :type image: List[List[int]]
        :type sr: int
        :type sc: int
        :type newColor: int
        :rtype: List[List[int]]
        """
        queue = [(sr, sc)]
        old_col, new_col = image[sr][sc], newColor
        if old_col == new_col: return image
        while queue:
            i, j = queue.pop()
            image[i][j] = new_col
            queue.extend([(x, y) for x, y in [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)] \
                        if 0 <= x < len(image) and 0 <= y < len(image[0]) and image[x][y] == old_col])
        return image


# ============================================================================

# 735. Asteroid Collision
# Difficulty: Medium
# link: https://leetcode.com/problems/asteroid-collision/
# Companies: Amazon,Lyft
# Categories: Stack

# ----------------------------------------------------------------------------

class Solution(object):
    def asteroidCollision(self, asteroids):
        """
        :type asteroids: List[int]
        :rtype: List[int]
        """
        stack = []
        while asteroids:
            stack.append(asteroids.pop())
            while len(stack) >= 2 and stack[-1] > 0 and stack[-2] < 0:
                a, b = stack.pop(), stack.pop()
                if abs(a) > abs(b):
                    stack.append(a)
                elif abs(a) < abs(b):
                    stack.append(b)
        return list(reversed(stack))


# ============================================================================

# 739. Daily Temperatures
# Difficulty: Medium
# link: https://leetcode.com/problems/daily-temperatures/
# Companies: Uber,Amazon,Google,Facebook,Apple
# Categories: Hash Table,Stack

# ----------------------------------------------------------------------------

class Solution(object):
    def dailyTemperatures(self, temperatures):
        """
        :type temperatures: List[int]
        :rtype: List[int]
        """
        res, stack = [], []
        for i in range(len(temperatures) - 1, -1, -1):
            while stack and temperatures[stack[-1]] <= temperatures[i]:
                stack.pop()
            res.append(stack[-1] - i if stack else 0)
            stack.append(i)
        res.reverse()
        return res


# ============================================================================

# 744. Find Smallest Letter Greater Than Target
# Difficulty: Easy
# link: https://leetcode.com/problems/find-smallest-letter-greater-than-target/
# Companies: LinkedIn
# Categories: Binary Search

# ----------------------------------------------------------------------------

class Solution(object):
    def nextGreatestLetter(self, letters, target):
        i, j = 0, len(letters)
        while i < j:
            mid = (i + j) / 2
            if letters[mid] == target: i = i + 1
            elif letters[mid] < target: i = mid + 1
            else: j = mid
        return letters[i % len(letters)]


# ============================================================================

# 746. Min Cost Climbing Stairs
# Difficulty: Easy
# link: https://leetcode.com/problems/min-cost-climbing-stairs/
# Companies: Amazon
# Categories: Array,Dynamic Programming

# ----------------------------------------------------------------------------

class Solution(object):
    def minCostClimbingStairs(self, cost):
        """
        :type cost: List[int]
        :rtype: int
        """
        if len(cost) <= 2: return int(bool(len(cost)))
        dp_cost_step = cost[:2]
        for i in range(2, len(cost)):
            dp_cost_step.append(min(dp_cost_step[-2:]) + cost[i])
        return min(dp_cost_step[-2:])


# ============================================================================

# 747. Largest Number At Least Twice of Others
# Difficulty: Easy
# link: https://leetcode.com/problems/largest-number-at-least-twice-of-others/
# Companies: Google
# Categories: Array

# ----------------------------------------------------------------------------

class Solution(object):
    def dominantIndex(self, nums):
        if len(nums) == 1: return 0
        i = j = None
        for idx, num in enumerate(nums):
            if j is None or num >= nums[j]: i, j = j, idx
            elif i is None or num >= nums[i]: i = idx
        return j if nums[j] >= nums[i] * 2 else -1


# ============================================================================

# 748. Shortest Completing Word
# Difficulty: Easy
# link: https://leetcode.com/problems/shortest-completing-word/
# Companies: Google,LinkedIn
# Categories: Hash Table

# ----------------------------------------------------------------------------

class Solution(object):
    def shortestCompletingWord(self, licensePlate, words):
        """
        :type licensePlate: str
        :type words: List[str]
        :rtype: str
        """
        from collections import Counter
        counts = Counter(char.lower() for char in licensePlate if char.isalpha())
        shortest_word = shortest_len = None
        for word in words:
            word_char_counts = Counter(word.lower())
            if (shortest_len is None or len(word) < shortest_len) and \
                all(count <= word_char_counts[plate_c] for plate_c, count in counts.iteritems()):
                shortest_word, shortest_len = word, len(word)
        return shortest_word


# ============================================================================

# 762. Prime Number of Set Bits in Binary Representation
# Difficulty: Easy
# link: https://leetcode.com/problems/prime-number-of-set-bits-in-binary-representation/
# Companies: Amazon
# Categories: Bit Manipulation

# ----------------------------------------------------------------------------

class Solution(object):
    def countPrimeSetBits(self, L, R):
        """
        :type L: int
        :type R: int
        :rtype: int
        """
        primes = {2, 3, 5, 7, 11, 13, 17, 19}
        return sum((bin(i).count('1') in primes) for i in range(L, R + 1))


# ============================================================================

# 763. Partition Labels
# Difficulty: Medium
# link: https://leetcode.com/problems/partition-labels/
# Companies: Amazon
# Categories: Two Pointers,Greedy

# ----------------------------------------------------------------------------

class Solution(object):
    def partitionLabels(self, S):
        """
        :type S: str
        :rtype: List[int]
        """
        from collections import Counter
        counts = Counter(S)
        i = j = 0
        res = []
        while i < len(S):
            seen = set([S[j]])
            while j < len(S) and seen:
                char = S[j]
                seen.add(char)
                counts[char] -= 1
                if not counts[char]: seen.remove(char)
                j += 1
            res.append(j - i)
            i = j
        return res


# ============================================================================

# 766. Toeplitz Matrix
# Difficulty: Easy
# link: https://leetcode.com/problems/toeplitz-matrix/
# Companies: Google,Facebook
# Categories: Array

# ----------------------------------------------------------------------------

class Solution(object):
    def isToeplitzMatrix(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: bool
        """
        m, n = len(matrix), len(matrix[0])
        for i in range(m):
            for j in range(n):
                if i and j and matrix[i - 1][j - 1] != matrix[i][j]:
                    return False
        return True


# ============================================================================

# 767. Reorganize String
# Difficulty: Medium
# link: https://leetcode.com/problems/reorganize-string/
# Companies: Twitch,Amazon,Google,Facebook,Microsoft
# Categories: String,Heap,Greedy,Sort

# ----------------------------------------------------------------------------

class Solution(object):
    def reorganizeString(self, S):
        """
        :type S: str
        :rtype: str
        """
        from collections import Counter
        counts = [(count, char) for char, count in Counter(S).iteritems()]
        max_freq, max_freq_char = max(counts)
        if max_freq > ((len(S) + 1)/ 2): return ""

        res = [[max_freq_char] for _ in range(max_freq)]
        i = 0
        while counts:
            count, char = counts.pop()
            if char != max_freq_char:
                for j in range(i, i + count):
                    res[j % max_freq].append(char)
                i += count
        return ''.join([''.join(x) for x in res])


# ============================================================================

# 771. Jewels and Stones
# Difficulty: Easy
# link: https://leetcode.com/problems/jewels-and-stones/
# Companies: Google,Apple
# Categories: Hash Table

# ----------------------------------------------------------------------------

class Solution(object):
    def numJewelsInStones(self, J, S):
        """
        :type J: str
        :type S: str
        :rtype: int
        """
        jewel_set = set(list(J))
        return sum(stone in jewel_set for stone in S)


# ============================================================================

# 781. Rabbits in Forest
# Difficulty: Medium
# link: https://leetcode.com/problems/rabbits-in-forest/
# Companies: Wish
# Categories: Hash Table,Math

# ----------------------------------------------------------------------------

class Solution(object):
    def numRabbits(self, answers):
        """
        :type answers: List[int]
        :rtype: int
        """
        from collections import Counter
        from math import ceil
        counts = Counter(x + 1 for x in answers)
        return int(sum(ceil(float(head_count) / quantity) * quantity
                   for quantity, head_count in counts.iteritems()))


# ============================================================================

# 783. Minimum Distance Between BST Nodes
# Difficulty: Easy
# link: https://leetcode.com/problems/minimum-distance-between-bst-nodes/
# Companies: Amazon,Google
# Categories: Tree,Recursion

# ----------------------------------------------------------------------------

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def minDiffInBST(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        self.prev, self.min_dif = float('-inf'), float('inf')
        def in_order(node):
            if not node: return
            in_order(node.left)
            self.min_dif = min(self.min_dif, node.val - self.prev)
            self.prev = node.val
            in_order(node.right)
        in_order(root)
        return self.min_dif


# ============================================================================

# 784. Letter Case Permutation
# Difficulty: Easy
# link: https://leetcode.com/problems/letter-case-permutation/
# Companies: Microsoft
# Categories: Backtracking,Bit Manipulation

# ----------------------------------------------------------------------------

class Solution(object):
    def letterCasePermutation(self, S):
        """
        :type S: str
        :rtype: List[str]
        """
        res = []
        S = list(S.lower())
        def _letterCasePermutation(i):
            if i == len(S):
                return res.append(''.join(S))
            elif S[i].isalpha():
                S[i] = S[i].upper()
                _letterCasePermutation(i + 1)
                S[i] = S[i].lower()
            _letterCasePermutation(i + 1)
        _letterCasePermutation(0)
        return res


# ============================================================================

# 788. Rotated Digits
# Difficulty: Easy
# link: https://leetcode.com/problems/rotated-digits/
# Companies: Google
# Categories: String

# ----------------------------------------------------------------------------

class Solution(object):
    def rotatedDigits(self, N):
        """
        :type N: int
        :rtype: int
        """
        from collections import defaultdict
        dig_map = [0, 1, 5, -1, -1, 2, 9, -1, 8, 6]
        dp = [1, 1, 2, 0, 0, 2, 2, 0, 1, 2] #0: can't flip, 1: same, 2: diff
        for i in range(10, N + 1):
            last_dig = i % 10
            first_part = i / 10
            if dp[last_dig] == 2 and dp[first_part] or dp[first_part] == 2 and dp[last_dig]:
                dp.append(2)
            elif dp[last_dig] == dp[first_part] == 1:
                dp.append(1)
            else:
                dp.append(0)
        return sum(i == 2 for i in dp[1:N+1])


# ============================================================================

# 789. Escape The Ghosts
# Difficulty: Medium
# link: https://leetcode.com/problems/escape-the-ghosts/
# Companies: Google
# Categories: Math

# ----------------------------------------------------------------------------

class Solution(object):
    def escapeGhosts(self, ghosts, target):
        """
        :type ghosts: List[List[int]]
        :type target: List[int]
        :rtype: bool
        """
        def dist(x, y): return sum(map(abs, [target[0] - x, target[1] - y]))
        dist_ghost = dist(0, 0)
        return not any(dist(x, y) <= dist_ghost
            for x, y in ghosts)


# ============================================================================

# 791. Custom Sort String
# Difficulty: Medium
# link: https://leetcode.com/problems/custom-sort-string/
# Companies: Uber,Amazon,Facebook
# Categories: String

# ----------------------------------------------------------------------------

class Solution(object):
    def customSortString(self, S, T):
        """
        :type S: str
        :type T: str
        :rtype: str
        """
        from collections import Counter
        counts = Counter(T)
        seq = S + ''.join(set(T)-set(S))
        return ''.join(char * counts[char] for char in seq)


# ============================================================================

# 796. Rotate String
# Difficulty: Easy
# link: https://leetcode.com/problems/rotate-string/
# Companies: Microsoft,Amazon,Apple,eBay
# Categories:

# ----------------------------------------------------------------------------

class Solution(object):
    def rotateString(self, A, B):
        """
        :type A: str
        :type B: str
        :rtype: bool
        """
        return not any([A, B]) or any(A[i:] + A[:i] == B for i in range(len(A)))


# ============================================================================

# 797. All Paths From Source to Target
# Difficulty: Medium
# link: https://leetcode.com/problems/all-paths-from-source-to-target/
# Companies: Bloomberg
# Categories:

# ----------------------------------------------------------------------------

class Solution(object):
    def allPathsSourceTarget(self, graph):
        paths, cur_path = [], []
        def dfs(cur):
            if cur == len(graph) - 1: paths.append(cur_path + [cur])
            else:
                cur_path.append(cur)
                for adj in graph[cur]: dfs(adj)
                cur_path.pop()
        dfs(0)
        return paths


# ============================================================================

# 804. Unique Morse Code Words
# Difficulty: Easy
# link: https://leetcode.com/problems/unique-morse-code-words/
# Companies: Apple
# Categories: String

# ----------------------------------------------------------------------------

class Solution(object):
    def uniqueMorseRepresentations(self, words):
        """
        :type words: List[str]
        :rtype: int
        """
        mapping = [".-","-...","-.-.","-..",".","..-.","--.",
                   "....","..",".---","-.-",".-..","--","-.",
                   "---",".--.","--.-",".-.","...","-","..-",
                   "...-",".--","-..-","-.--","--.."]
        def covert_to_morse(word):
            return ''.join(mapping[ord(char) - ord('a')] for char in word.lower())
        return len(set(map(covert_to_morse, words)))


# ============================================================================

# 806. Number of Lines To Write String
# Difficulty: Easy
# link: https://leetcode.com/problems/number-of-lines-to-write-string/
# Companies: Google
# Categories:

# ----------------------------------------------------------------------------

class Solution(object):
    def numberOfLines(self, widths, S):
        """
        :type widths: List[int]
        :type S: str
        :rtype: List[int]
        """
        num_lines = line_width = 0
        for i, char in enumerate(S):
            width = widths[ord(char) - ord('a')]
            if not num_lines or line_width + width > 100:
                line_width = width
                num_lines += 1
            else:
                line_width += width
        return (num_lines, line_width)


# ============================================================================

# 807. Max Increase to Keep City Skyline
# Difficulty: Medium
# link: https://leetcode.com/problems/max-increase-to-keep-city-skyline/
# Companies: Amazon
# Categories:

# ----------------------------------------------------------------------------

class Solution(object):
    def maxIncreaseKeepingSkyline(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        m, n = len(grid), len(grid[0])
        hor_view, ver_view = [0] * m, [0] * n
        for i in range(m):
            for j in range(n):
                hor_view[i] = max(hor_view[i], grid[i][j])
                ver_view[j] = max(ver_view[j], grid[i][j])
        total = sum(
            min(hor_view[i], ver_view[j]) - grid[i][j]
            for i in range(m) for j in range(n)
        )
        return total


# ============================================================================

# 811. Subdomain Visit Count
# Difficulty: Easy
# link: https://leetcode.com/problems/subdomain-visit-count/
# Companies: Roblox,Indeed,Wayfair,Karat,Pinterest
# Categories: Hash Table

# ----------------------------------------------------------------------------

class Solution(object):
    def subdomainVisits(self, cpdomains):
        """
        :type cpdomains: List[str]
        :rtype: List[str]
        """
        from collections import defaultdict
        subdomain_counts = defaultdict(int)
        for cpdomain in cpdomains:
            count, domain = cpdomain.split(' ')
            domain = domain.split('.')
            for i in range(len(domain)):
                subdomain_counts['.'.join(domain[i:])] += int(count)
        return ["%d %s" %(count, domain) for domain, count in subdomain_counts.iteritems()]


# ============================================================================

# 814. Binary Tree Pruning
# Difficulty: Medium
# link: https://leetcode.com/problems/binary-tree-pruning/
# Companies: Capital One
# Categories: Tree

# ----------------------------------------------------------------------------

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def pruneTree(self, root):
        """
        :type root: TreeNode
        :rtype: TreeNode
        """
        def _pruneTree(node):
            if not node: return True
            l, r =  _pruneTree(node.left), _pruneTree(node.right)
            if l: node.left = None
            if r: node.right = None
            return not node.val and l and r
        _pruneTree(root)
        return root


# ============================================================================

# 817. Linked List Components
# Difficulty: Medium
# link: https://leetcode.com/problems/linked-list-components/
# Companies: Google
# Categories: Linked List

# ----------------------------------------------------------------------------

# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def numComponents(self, head, G):
        """
        :type head: ListNode
        :type G: List[int]
        :rtype: int
        """
        G = set(G)
        cur = head
        prev_in_G = False
        total = 0
        while cur:
            if not prev_in_G and cur.val in G:
                total += 1
            prev_in_G = cur.val in G
            cur = cur.next
        return total


# ============================================================================

# 819. Most Common Word
# Difficulty: Easy
# link: https://leetcode.com/problems/most-common-word/
# Companies: Amazon
# Categories: String

# ----------------------------------------------------------------------------

class Solution(object):
    def mostCommonWord(self, paragraph, banned):
        """
        :type paragraph: str
        :type banned: List[str]
        :rtype: str
        """
        from collections import Counter

        counts = Counter((''.join([char for char in paragraph if char.isalpha() or char == ' '])).lower().split(' '))
        freqs = sorted([(freq, word) for word, freq in counts.iteritems()], reverse=True)
        return next((word for freq, word in freqs if word not in banned), None)


# ============================================================================

# 820. Short Encoding of Words
# Difficulty: Medium
# link: https://leetcode.com/problems/short-encoding-of-words/
# Companies:
# Categories:

# ----------------------------------------------------------------------------

class Solution(object):
    def minimumLengthEncoding(self, words):

        words = [word[::-1] for word in words]
        tree = {}
        for word in words:
            cur = tree
            for char in word:
                cur = cur.setdefault(char, {})
        self.res = 0
        def dfs(depth, cur):
            if not cur:
                self.res += depth + 1
                return
            for adj in cur:
                dfs(depth + 1, cur[adj])
        dfs(0, tree)
        return self.res


# ============================================================================

# 821. Shortest Distance to a Character
# Difficulty: Easy
# link: https://leetcode.com/problems/shortest-distance-to-a-character/
# Companies: Google,Atlassian
# Categories:

# ----------------------------------------------------------------------------

class Solution(object):
    def shortestToChar(self, S, C):
        """
        :type S: str
        :type C: str
        :rtype: List[int]
        """
        res = []
        last_C = float('-inf')
        for i, char in enumerate(S):
            if char == C:
                last_C = i
            res.append(i - last_C)
        last_C = float('inf')
        for i in range(len(S) -1, -1, -1):
            char = S[i]
            if char == C:
                last_C = i
            res[i] = min(res[i], last_C - i)
        return res


# ============================================================================

# 824. Goat Latin
# Difficulty: Easy
# link: https://leetcode.com/problems/goat-latin/
# Companies: Facebook
# Categories: String

# ----------------------------------------------------------------------------

class Solution(object):
    def toGoatLatin(self, S):
        """
        :type S: str
        :rtype: str
        """
        def parse_prefix(word):
            return word if (word[0].lower() in 'aeiou') else (word[1:] + word[0])
        return ' '.join([word + 'ma' + ((i + 1) * 'a')
                         for i, word in enumerate(map(parse_prefix, S.split()))])


# ============================================================================

# 830. Positions of Large Groups
# Difficulty: Easy
# link: https://leetcode.com/problems/positions-of-large-groups/
# Companies: Google
# Categories: Array

# ----------------------------------------------------------------------------

class Solution(object):
    def largeGroupPositions(self, S):
        res = []
        prev = 0
        for i in range(len(S) + 1):
            if i == 0: continue
            if i == len(S) or S[i] != S[i - 1]:
                if i - prev >= 3: res.append([prev, i - 1])
                prev = i
        return res


# ============================================================================

# 832. Flipping an Image
# Difficulty: Easy
# link: https://leetcode.com/problems/flipping-an-image/
# Companies: Google
# Categories: Array

# ----------------------------------------------------------------------------

class Solution(object):
    def flipAndInvertImage(self, A):
        """
        :type A: List[List[int]]
        :rtype: List[List[int]]
        """
        return [map(lambda x: 1 - x, reversed(lst)) for lst in A]


# ============================================================================

# 836. Rectangle Overlap
# Difficulty: Easy
# link: https://leetcode.com/problems/rectangle-overlap/
# Companies: Google,Adobe,Microsoft
# Categories: Math

# ----------------------------------------------------------------------------

class Solution(object):
    def isRectangleOverlap(self, rec1, rec2):
        """
        :type rec1: List[int]
        :type rec2: List[int]
        :rtype: bool
        """
        recs = zip(rec1, rec2)
        (x1, y1), (x2, y2) = map(max, recs[:2]), map(min, recs[2:])
        return x1 < x2 and y1 < y2


# ============================================================================

# 840. Magic Squares In Grid
# Difficulty: Easy
# link: https://leetcode.com/problems/magic-squares-in-grid/
# Companies: Wayfair
# Categories: Array

# ----------------------------------------------------------------------------

class Solution(object):
    def numMagicSquaresInside(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        def is_magic(i, j):
            qwe, asd, zxc = [grid[i + k][j:j + 3] for k in range(3)]
            (q,w,e), (a,s,d), (z,x,c) = qwe, asd, zxc
            qaz, wsx, edc = (q,a,z),(w,s,x),(e,d,c)
            qsc, esz = (q,s,c),(e,s,z)
            totals = map(sum,[qwe, asd, zxc, qaz, wsx, edc, qsc, esz])
            return {q,w,e,a,s,d,z,x,c} == {1,2,3,4,5,6,7,8,9} and \
                    all(totals[i] == totals[i - 1] for i in range(1, len(totals)))

        return sum(is_magic(i, j) for i in range(len(grid) - 2) for j in range(len(grid[0]) - 2))


# ============================================================================

# 841. Keys and Rooms
# Difficulty: Medium
# link: https://leetcode.com/problems/keys-and-rooms/
# Companies: Google
# Categories: Depth-first Search,Graph

# ----------------------------------------------------------------------------

class Solution(object):
    def canVisitAllRooms(self, rooms):
        """
        :type rooms: List[List[int]]
        :rtype: bool
        """
        if not rooms: return True
        visited, bfs = {0}, {0}
        while bfs:
            bfs = {next_room
                   for room in bfs
                   for next_room in rooms[room]
                   if next_room not in visited and (visited.add(next_room) is None)}
        return len(visited) == len(rooms)


# ============================================================================

# 844. Backspace String Compare
# Difficulty: Easy
# link: https://leetcode.com/problems/backspace-string-compare/
# Companies: Google,Atlassian
# Categories: Two Pointers,Stack

# ----------------------------------------------------------------------------

class Solution(object):
    def backspaceCompare(self, S, T):
        cnt_i = cnt_j = 0
        i, j = len(S) - 1, len(T) - 1

        while True:
            if i == -1 and j == -1: return True
            elif i >= 0 and S[i] == "#":
                cnt_i += 1
                i -= 1
            elif j >= 0 and T[j] == "#":
                cnt_j += 1
                j -= 1
            elif cnt_j and j > -1:
                cnt_j -= 1
                j -= 1
            elif cnt_i and i > -1:
                cnt_i -= 1
                i -= 1
            else:
                if i == -1 or j == -1 or S[i] != T[j]: return False
                i -= 1
                j -= 1



    def _backspaceCompare(self, S, T):
        def process_str(string):
            res = []
            for c in string:
                if c != '#':
                    res.append(c)
                elif res:
                    res.pop()
            return res
        return process_str(S) == process_str(T)


# ============================================================================

# 848. Shifting Letters
# Difficulty: Medium
# link: https://leetcode.com/problems/shifting-letters/
# Companies:
# Categories: String

# ----------------------------------------------------------------------------

class Solution(object):
    def shiftingLetters(self, S, shifts):
        """
        :type S: str
        :type shifts: List[int]
        :rtype: str
        """
        cur = 0
        for i in range(len(shifts) -1, -1, -1):
            shifts[i] += cur
            cur = shifts[i]
        return ''.join(chr((ord(S[i]) - ord('a') + shifts[i]) % 26 + ord('a'))
                       for i in range(len(shifts)))


# ============================================================================

# 849. Maximize Distance to Closest Person
# Difficulty: Easy
# link: https://leetcode.com/problems/maximize-distance-to-closest-person/
# Companies: Google
# Categories: Array

# ----------------------------------------------------------------------------

class Solution(object):
    def maxDistToClosest(self, seats):
        prev = None
        res = 0
        for i, seated in enumerate(seats):
            if seated:
                if prev is None: res = i
                else: res = max(res, (i - prev) / 2)
                prev = i
        return max(res, len(seats) - prev - 1)


# ============================================================================

# 852. Peak Index in a Mountain Array
# Difficulty: Easy
# link: https://leetcode.com/problems/peak-index-in-a-mountain-array/
# Companies: Google
# Categories: Binary Search

# ----------------------------------------------------------------------------

class Solution(object):
    def peakIndexInMountainArray(self, A):
        i, j = 0, len(A)
        A_ = lambda i: A[i] if 0 <= i < len(A) else float('-inf')
        while i < j:
            mid = (i + j) / 2
            if A_(mid) > A_(mid - 1) and A_(mid) > A_(mid + 1): return mid
            elif A_(mid - 1) < A_(mid) < A_(mid + 1): i = mid + 1
            elif A_(mid - 1) > A_(mid) > A_(mid + 1): j = mid


# ============================================================================

# 853. Car Fleet
# Difficulty: Medium
# link: https://leetcode.com/problems/car-fleet/
# Companies: Google
# Categories: Sort

# ----------------------------------------------------------------------------

class Solution(object):
    def carFleet(self, target, position, speed):
        """
        :type target: int
        :type position: List[int]
        :type speed: List[int]
        :rtype: int
        """
        pos_time = sorted([p, float(target - p) / s] for p, s in zip(position, speed))
        cnt = slowest = 0
        for _, t in reversed(pos_time):
            if t > slowest:
                cnt += 1
                slowest = t
        return cnt


# ============================================================================

# 856. Score of Parentheses
# Difficulty: Medium
# link: https://leetcode.com/problems/score-of-parentheses/
# Companies:
# Categories: String,Stack

# ----------------------------------------------------------------------------

class Solution(object):
    def scoreOfParentheses(self, S):
        """
        :type S: str
        :rtype: int
        """
        # True is open, False is close
        points = [0] * (len(S) + 1)
        cur_lvl = 0
        for char in S:
            if char == '(':
                cur_lvl += 1
            elif char == ')':
                cur_lvl -= 1
                if points[cur_lvl + 1]:
                    points[cur_lvl] += points[cur_lvl + 1] * 2
                    points[cur_lvl + 1] = 0
                else:
                    points[cur_lvl] += 1
        return points[0]


# ============================================================================

# 859. Buddy Strings
# Difficulty: Easy
# link: https://leetcode.com/problems/buddy-strings/
# Companies: Google
# Categories: String

# ----------------------------------------------------------------------------

class Solution(object):
    def buddyStrings(self, A, B):
        """
        :type A: str
        :type B: str
        :rtype: bool
        """
        from collections import Counter
        if len(A) != len(B): return False
        l = next((i for i in range(len(A)) if A[i] != B[i]), None)
        r = next((i for i in range(len(A) -1, -1, -1) if A[i] != B[i]), None)
        if l is None: return len(set(A)) != len(A)
        return A[:l] + A[r] + A[l + 1:r] + A[l] + A[r + 1:] == B


# ============================================================================

# 860. Lemonade Change
# Difficulty: Easy
# link: https://leetcode.com/problems/lemonade-change/
# Companies: Atlassian
# Categories: Greedy

# ----------------------------------------------------------------------------

class Solution(object):
    def lemonadeChange(self, bills):
        """
        :type bills: List[int]
        :rtype: bool
        """
        cnt = {i: 0 for i in [20, 10, 5]}
        for b in bills:
            total_change = b - 5
            cnt[b] += 1
            for c in [20, 10, 5]:
                num = min(total_change / c, cnt[c]) if (cnt[c] > 0) else 0
                total_change -= num * c
                cnt[c] -= num
            if total_change: return False
        return True


# ============================================================================

# 861. Score After Flipping Matrix
# Difficulty: Medium
# link: https://leetcode.com/problems/score-after-flipping-matrix/
# Companies: IIT Bombay
# Categories: Greedy

# ----------------------------------------------------------------------------

class Solution(object):
    def matrixScore(self, A):
        """
        :type A: List[List[int]]
        :rtype: int
        """
        m, n = len(A), len(A[0]) if A else 0

        def get_score():
            return sum(reduce(lambda x, bit: x << 1 | bit, row, 0) for row in A)

        for r in range(m):
            if not A[r][0]:
                for c in range(n): A[r][c] = not A[r][c]

        for c in range(1, n):
            if sum(A[r][c] for r in range(m)) <= (m / 2):
                for r in range(m): A[r][c] = not A[r][c]

        return get_score()


# ============================================================================

# 863. All Nodes Distance K in Binary Tree
# Difficulty: Medium
# link: https://leetcode.com/problems/all-nodes-distance-k-in-binary-tree/
# Companies: Uber,Oracle,Amazon,Facebook
# Categories: Tree,Depth-first Search,Breadth-first Search

# ----------------------------------------------------------------------------

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def distanceK(self, root, target, K):
        """
        :type root: TreeNode
        :type target: TreeNode
        :type K: int
        :rtype: List[int]
        """
        child_to_parent = {}
        def dfs(parent, node):
            if not node: return
            child_to_parent[node] = parent
            dfs(node, node.left)
            dfs(node, node.right)
        dfs(None, root)
        visited = set([target])
        bfs = [target]
        for _ in range(K):
            bfs = [child
                   for node in bfs
                   for child in [child_to_parent[node], node.left, node.right]
                   if child and child not in visited and
                   (visited.add(child) is None) # add it to visited
                  ]
        return [node.val for node in bfs]


# ============================================================================

# 865. Smallest Subtree with all the Deepest Nodes
# Difficulty: Medium
# link: https://leetcode.com/problems/smallest-subtree-with-all-the-deepest-nodes/
# Companies: Facebook
# Categories: Tree

# ----------------------------------------------------------------------------

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def subtreeWithAllDeepest(self, root):
        """
        :type root: TreeNode
        :rtype: TreeNode
        """
        bfs = {root}
        while bfs:
            prev = bfs
            bfs = {kid for node in bfs for kid in [node.left, node.right] if kid}
        deepest = prev
        def dfs_deepest(node):
            if not node or node in deepest: return node
            l, r = dfs_deepest(node.left), dfs_deepest(node.right)
            return node if l and r else l or r
        return dfs_deepest(root)


# ============================================================================

# 867. Transpose Matrix
# Difficulty: Easy
# link: https://leetcode.com/problems/transpose-matrix/
# Companies: ServiceNow
# Categories: Array

# ----------------------------------------------------------------------------

class Solution(object):
    def transpose(self, A):
        """
        :type A: List[List[int]]
        :rtype: List[List[int]]
        """
        return [[A[i][j] for i in range(len(A))]
                for j in range(len(A[0]))] or [[]]


# ============================================================================

# 884. Uncommon Words from Two Sentences
# Difficulty: Easy
# link: https://leetcode.com/problems/uncommon-words-from-two-sentences/
# Companies: Expedia,Microsoft
# Categories: Hash Table

# ----------------------------------------------------------------------------

class Solution(object):
    def uncommonFromSentences(self, A, B):
        """
        :type A: str
        :type B: str
        :rtype: List[str]
        """
        from collections import Counter
        A = A.split(' ')
        B = B.split(' ')
        A = Counter(A)
        B = Counter(B)
        return [a for a, count in A.iteritems() if count == 1 and a not in B] + \
            [b for b, count in B.iteritems() if count == 1 and b not in A]


# ============================================================================

# 888. Fair Candy Swap
# Difficulty: Easy
# link: https://leetcode.com/problems/fair-candy-swap/
# Companies: Fidessa
# Categories: Array

# ----------------------------------------------------------------------------

class Solution(object):
    def fairCandySwap(self, A, B):
        """
        :type A: List[int]
        :type B: List[int]
        :rtype: List[int]
        """
        n, m = sum(A), sum(B)
        gap = (m - n) / 2
        B_set = set(B)
        for i in A:
            if (i + gap) in B_set:
                return (i, (i + gap))


# ============================================================================

# 892. Surface Area of 3D Shapes
# Difficulty: Easy
# link: https://leetcode.com/problems/surface-area-of-3d-shapes/
# Companies:
# Categories: Math,Geometry

# ----------------------------------------------------------------------------

class Solution(object):
    def surfaceArea(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        n = len(grid)
        def surface(n): return n * 6 - (n - 1) * 2 if n else 0
        total = sum(surface(grid[i][j]) for i in range(n) for j in range(n))
        adj_ver = sum(min(grid[i][j], grid[i][j + 1]) * 2 for i in range(n) for j in range(n - 1))
        adj_hor = sum(min(grid[i][j], grid[i + 1][j ]) * 2 for i in range(n - 1) for j in range(n))
        return total - adj_ver - adj_hor


# ============================================================================

# 894. All Possible Full Binary Trees
# Difficulty: Medium
# link: https://leetcode.com/problems/all-possible-full-binary-trees/
# Companies: Google
# Categories: Tree,Recursion

# ----------------------------------------------------------------------------

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def allPossibleFBT(self, N):
        """
        :type N: int
        :rtype: List[TreeNode]
        """

        def construct(n):
            if n == 1: return [TreeNode(0)]
            trees = []
            for k in range(1, n, 2):
                left, right = construct(k), construct(n - k - 1)
                for l in left:
                    for r in right:
                        node = TreeNode(0)
                        node.left, node.right = l, r
                        trees.append(node)
            return trees
        return construct(N)


# ============================================================================

# 896. Monotonic Array
# Difficulty: Easy
# link: https://leetcode.com/problems/monotonic-array/
# Companies: Facebook
# Categories: Array

# ----------------------------------------------------------------------------

class Solution(object):
    def isMonotonic(self, A):
        """
        :type A: List[int]
        :rtype: bool
        """
        if len(A) <= 2: return True
        return all(A[i] <= A[i + 1] for i in range(len(A) - 1)) or \
               all(A[i] >= A[i + 1] for i in range(len(A) - 1))


# ============================================================================

# 897. Increasing Order Search Tree
# Difficulty: Easy
# link: https://leetcode.com/problems/increasing-order-search-tree/
# Companies:
# Categories: Tree,Depth-first Search

# ----------------------------------------------------------------------------

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def increasingBST(self, root):
        """
        :type root: TreeNode
        :rtype: TreeNode
        """
        self.cur = dummy = TreeNode('dummy')
        def create_tree(node):
            if not node: return
            if node.left:
                create_tree(node.left)
            self.cur.right = TreeNode(node.val)
            self.cur = self.cur.right
            if node.right:
                create_tree(node.right)
        create_tree(root)
        return dummy.right


# ============================================================================

# 908. Smallest Range I
# Difficulty: Easy
# link: https://leetcode.com/problems/smallest-range-i/
# Companies: Adobe
# Categories: Math

# ----------------------------------------------------------------------------

class Solution(object):
    def smallestRangeI(self, A, K):
        """
        :type A: List[int]
        :type K: int
        :rtype: int
        """
        min_a = min(A)
        max_a = max(A)
        return max(max_a - min_a - 2*K, 0)


# ============================================================================

# 911. Online Election
# Difficulty: Medium
# link: https://leetcode.com/problems/online-election/
# Companies: Google,Apple
# Categories: Binary Search

# ----------------------------------------------------------------------------

import bisect
class TopVotedCandidate(object):

    def __init__(self, persons, times):
        """
        :type persons: List[int]
        :type times: List[int]
        """

        person_to_count = {}
        max_vote = 0
        winning = self.winning = []
        self.times = times
        for i in range(len(persons)):
            person_to_count[persons[i]] = person_to_count.get(persons[i], 0) + 1
            if person_to_count[persons[i]] >= max_vote:
                max_vote = person_to_count[persons[i]]
                winning.append(persons[i])
            else:
                winning.append(winning[-1])

    def q(self, t):
        """
        :type t: int
        :rtype: int
        """
        idx = bisect.bisect_left(self.times, t)
        if idx >= len(self.times) or self.times[idx] > t: idx -= 1
        return self.winning[idx]



# Your TopVotedCandidate object will be instantiated and called as such:
# obj = TopVotedCandidate(persons, times)
# param_1 = obj.q(t)


# ============================================================================

# 912. Sort an Array
# Difficulty: Medium
# link: https://leetcode.com/problems/sort-an-array/
# Companies: Apple
# Categories:

# ----------------------------------------------------------------------------

class Solution(object):
    def sortArray(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        return sorted(nums)


# ============================================================================

# 914. X of a Kind in a Deck of Cards
# Difficulty: Easy
# link: https://leetcode.com/problems/x-of-a-kind-in-a-deck-of-cards/
# Companies: Google
# Categories: Array,Math

# ----------------------------------------------------------------------------

class Solution(object):
    def hasGroupsSizeX(self, deck):
        """
        :type deck: List[int]
        :rtype: bool
        """
        from collections import Counter
        counts = Counter(deck)
        counts = set(counts.values())
        min_c = min(counts)
        return any( all((c % i == 0) for c in counts) for i in range(2, min_c + 1)) and min_c > 1


# ============================================================================

# 915. Partition Array into Disjoint Intervals
# Difficulty: Medium
# link: https://leetcode.com/problems/partition-array-into-disjoint-intervals/
# Companies: Grab
# Categories: Array

# ----------------------------------------------------------------------------

class Solution(object):
    def partitionDisjoint(self, A):
        """
        :type A: List[int]
        :rtype: int
        """
        r_min = [A[-1]]
        for i in range(len(A) -2, -1, -1):
            r_min.append(min(A[i], r_min[-1]))
        r_min.reverse()

        max_so_far = A[0]
        for i in range(len(A) - 1):
            if max_so_far <=  r_min[i + 1]:
                return i + 1
            max_so_far = max(max_so_far, A[i])
        return len(r_min)


# ============================================================================

# 916. Word Subsets
# Difficulty: Medium
# link: https://leetcode.com/problems/word-subsets/
# Companies: Google
# Categories: String

# ----------------------------------------------------------------------------

class Solution(object):
    def wordSubsets(self, A, B):
        """
        :type A: List[str]
        :type B: List[str]
        :rtype: List[str]
        """
        from collections import Counter

        max_c = {}
        for w2 in B:
            for char, count in Counter(w2).iteritems():
                max_c[char] = max(max_c.get(char, 0), count)

        res = []
        for w1 in A:
            c_word1 = Counter(w1)
            if all(char in c_word1 and c_word1[char] >= max_c[char] for char in max_c):
                res.append(w1)
        return res


# ============================================================================

# 925. Long Pressed Name
# Difficulty: Easy
# link: https://leetcode.com/problems/long-pressed-name/
# Companies:
# Categories: Two Pointers,String

# ----------------------------------------------------------------------------

class Solution(object):
    def isLongPressedName(self, name, typed):
        """
        :type name: str
        :type typed: str
        :rtype: bool
        """
        i, j = 0, 0
        while True:
            if i == len(name) and j == len(typed):
                return True
            elif i < len(name) and j < len(typed) and name[i] == typed[j]:
                i += 1
                j += 1
            elif j < len(typed) and name[i - 1] == typed[j]:
                j += 1
            else:
                return False


# ============================================================================

# 926. Flip String to Monotone Increasing
# Difficulty: Medium
# link: https://leetcode.com/problems/flip-string-to-monotone-increasing/
# Companies: Google
# Categories: Array

# ----------------------------------------------------------------------------

class Solution(object):
    def minFlipsMonoIncr(self, S):
        A = [0] # ones so far
        for char in S: A.append(A[-1] + bool(char=="1"))
        return min(A[i] + (len(S) - i - (A[-1] - A[i]))
                   for i in range(len(A)))


# ============================================================================

# 929. Unique Email Addresses
# Difficulty: Easy
# link: https://leetcode.com/problems/unique-email-addresses/
# Companies: Google
# Categories: String

# ----------------------------------------------------------------------------

class Solution(object):
    def numUniqueEmails(self, emails):
        """
        :type emails: List[str]
        :rtype: int
        """
        res = set()
        for email in emails:
            email = email.split("@")
            email[0] = email[0].replace(".", "")
            email[0] = email[0][:email[0].find("+")]
            res.add(tuple(email))

        return len(res)


# ============================================================================

# 930. Binary Subarrays With Sum
# Difficulty: Medium
# link: https://leetcode.com/problems/binary-subarrays-with-sum/
# Companies: C3 IoT
# Categories: Hash Table,Two Pointers

# ----------------------------------------------------------------------------

class Solution(object):
    def numSubarraysWithSum(self, A, S):
        cnt = {0: 1}
        res = sum_so_far = 0
        for dig in A:
            sum_so_far += dig
            res += cnt.get(sum_so_far - S, 0)
            cnt[sum_so_far] = cnt.get(sum_so_far, 0) + 1
        return res


# ============================================================================

# 931. Minimum Falling Path Sum
# Difficulty: Medium
# link: https://leetcode.com/problems/minimum-falling-path-sum/
# Companies: Google,Goldman Sachs
# Categories: Dynamic Programming

# ----------------------------------------------------------------------------

class Solution(object):
    def minFallingPathSum(self, A):
        """
        :type A: List[List[int]]
        :rtype: int
        """

        prev = A[0][:]
        row = []
        n = len(A)
        for i in range(1, n):
            row = []
            for j in range(n):
                cur = min((prev[j - 1] if (j - 1) >= 0 else float('inf')),
                          (prev[j + 1]  if (j + 1) < n else float('inf')),
                          prev[j])
                row.append(cur + A[i][j])
            prev = row
        return min(row or prev)


# ============================================================================

# 938. Range Sum of BST
# Difficulty: Easy
# link: https://leetcode.com/problems/range-sum-of-bst/
# Companies: Amazon,Facebook,Microsoft,Apple
# Categories: Tree,Recursion

# ----------------------------------------------------------------------------

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def rangeSumBST(self, root, L, R):
        """
        :type root: TreeNode
        :type L: int
        :type R: int
        :rtype: int
        """
        def _rangeSumBST(root):
            if not root: return 0
            elif root.val < L: return _rangeSumBST(root.right)
            elif R < root.val:  return _rangeSumBST(root.left)
            else: return _rangeSumBST(root.right) + _rangeSumBST(root.left) + root.val
        return _rangeSumBST(root)


# ============================================================================

# 944. Delete Columns to Make Sorted
# Difficulty: Easy
# link: https://leetcode.com/problems/delete-columns-to-make-sorted/
# Companies: Google
# Categories: Greedy

# ----------------------------------------------------------------------------

class Solution(object):
    def minDeletionSize(self, A):
        return sum(any(A[row][col] < A[row - 1][col]
                       for row in range(1, len(A)))
                   for col in range(len(A[0])))


# ============================================================================

# 949. Largest Time for Given Digits
# Difficulty: Easy
# link: https://leetcode.com/problems/largest-time-for-given-digits/
# Companies: LiveRamp
# Categories: Math

# ----------------------------------------------------------------------------

class Solution(object):
    def largestTimeFromDigits(self, A):
        """
        :type A: List[int]
        :rtype: str
        """
        valid_time = [((a * 10 + b), (c * 10 + d ), "%s%s:%s%s"%(a, b, c, d))
                        for i1, a in enumerate(A)
                        for i2, b in enumerate(A)
                        for i3, c in enumerate(A)
                        for i4, d in enumerate(A)
                        if len(set([i1,i2,i3,i4])) == 4 and (a * 10 + b) <= 23 and (c * 10 + d ) < 60]
        return max(valid_time)[2] if valid_time else ""


# ============================================================================

# 951. Flip Equivalent Binary Trees
# Difficulty: Medium
# link: https://leetcode.com/problems/flip-equivalent-binary-trees/
# Companies: Google
# Categories: Tree

# ----------------------------------------------------------------------------

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def flipEquiv(self, root1, root2):
        """
        :type root1: TreeNode
        :type root2: TreeNode
        :rtype: bool
        """

        def _flipEquiv(l, r):
            if not any([l, r]):
                return True
            elif not all([l, r]) or l.val != r.val:
                return False
            return _flipEquiv(l.left, r.right) and  _flipEquiv(l.right, r.left) or \
                 _flipEquiv(l.left, r.left) and  _flipEquiv(l.right, r.right)
        return _flipEquiv(root1, root2)


# ============================================================================

# 953. Verifying an Alien Dictionary
# Difficulty: Easy
# link: https://leetcode.com/problems/verifying-an-alien-dictionary/
# Companies: Facebook,Microsoft
# Categories: Hash Table

# ----------------------------------------------------------------------------

class Solution(object):
    def isAlienSorted(self, words, order):
        """
        :type words: List[str]
        :type order: str
        :rtype: bool
        """
        char_to_idx = {char: i for i, char in enumerate(order)}
        hashed_words = [[char_to_idx[char] for char in word] for word in words]
        return all(hashed_words[i] < hashed_words[i + 1] for i in range(len(hashed_words) - 1))


# ============================================================================

# 958. Check Completeness of a Binary Tree
# Difficulty: Medium
# link: https://leetcode.com/problems/check-completeness-of-a-binary-tree/
# Companies: Facebook,Microsoft,ByteDance
# Categories: Tree

# ----------------------------------------------------------------------------

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def isCompleteTree(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        if not root.left and not root.right:
            return True
        prev = None
        cur = [root]
        lvl = 1
        while cur:
            new_lvl = [child for node in cur for child in [node.left, node.right] if child]
            if not new_lvl:
                par = (len(cur) -1) / 2
                if cur[-1] == [prev[par].left, prev[par].right][(len(cur) - 1) % 2]:
                    return True
                return False
            elif len(cur) != lvl:
                return False
            prev = cur
            cur = new_lvl
            lvl *= 2
        return True


# ============================================================================

# 959. Regions Cut By Slashes
# Difficulty: Medium
# link: https://leetcode.com/problems/regions-cut-by-slashes/
# Companies: Uber
# Categories: Depth-first Search,Union Find,Graph

# ----------------------------------------------------------------------------

class Solution(object):
    def regionsBySlashes(self, grid):
        """
        :type grid: List[str]
        :rtype: int
        """
        transformed_grid = [([0] * (len(grid) * 3)) for _ in range(len(grid) * 3)]
        for i, row in enumerate(grid):
            for j, char in enumerate(row):
                a, b = i * 3, j * 3
                if char == "/":
                    transformed_grid[a][b + 2] = transformed_grid[a + 1][b + 1] = transformed_grid[a + 2][b] = 1
                elif char == "\\":
                    transformed_grid[a][b] = transformed_grid[a + 1][b + 1] = transformed_grid[a + 2][b + 2] = 1

        def count_island(i, j):
            if not (0 <= i < len(transformed_grid)) or not (0 <= j < len(transformed_grid)):
                return 0
            elif transformed_grid[i][j] == 0:
                transformed_grid[i][j] = 1
                for a, b in [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]:
                    count_island(a, b)
                return 1
            return 0
        count = 0
        for x in range(len(transformed_grid)):
            for y in range(len(transformed_grid)):
                count += count_island(x, y)
        return count


# ============================================================================

# 961. N-Repeated Element in Size 2N Array
# Difficulty: Easy
# link: https://leetcode.com/problems/n-repeated-element-in-size-2n-array/
# Companies: akamai
# Categories: Hash Table

# ----------------------------------------------------------------------------

class Solution(object):
    def repeatedNTimes(self, A):
        """
        :type A: List[int]
        :rtype: int
        """
        from collections import Counter
        return next((char for char, c in Counter(A).iteritems() if c == len(A) / 2))


# ============================================================================

# 962. Maximum Width Ramp
# Difficulty: Medium
# link: https://leetcode.com/problems/maximum-width-ramp/
# Companies: Google
# Categories: Array

# ----------------------------------------------------------------------------

class Solution(object):
    def maxWidthRamp(self, A):
        A = sorted(range(len(A)), key=A.__getitem__)
        res = 0
        min_idx_so_far = len(A)
        for i in A:
            res = max(res, i - min_idx_so_far)
            min_idx_so_far = min(min_idx_so_far, i)
        return res


# ============================================================================

# 977. Squares of a Sorted Array
# Difficulty: Easy
# link: https://leetcode.com/problems/squares-of-a-sorted-array/
# Companies: Uber,Google,Apple,Amazon,Facebook,Microsoft
# Categories: Array,Two Pointers

# ----------------------------------------------------------------------------

class Solution(object):
    def sortedSquares(self, A):
        """
        :type A: List[int]
        :rtype: List[int]
        """
        return sorted(map(lambda x:x**2, A))


# ============================================================================

# 979. Distribute Coins in Binary Tree
# Difficulty: Medium
# link: https://leetcode.com/problems/distribute-coins-in-binary-tree/
# Companies: Amazon,Google,Microsoft
# Categories: Tree,Depth-first Search

# ----------------------------------------------------------------------------

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def distributeCoins(self, root):
        total = [0]
        def dfs_sum(node):
            if not node: return 0
            res = dfs_sum(node.left) + dfs_sum(node.right) + node.val - 1
            total[0] += abs(res)
            return res
        dfs_sum(root)
        return total[0]


# ============================================================================

# 1002. Find Common Characters
# Difficulty: Easy
# link: https://leetcode.com/problems/find-common-characters/
# Companies: TripAdvisor
# Categories: Array,Hash Table

# ----------------------------------------------------------------------------

class Solution(object):
    def commonChars(self, A):
        from collections import Counter
        A = map(Counter, A)
        common = reduce(lambda lst_x, x:dict((key, min(x.get(key, 0), lst_x.get(key, 0))) for key in x), A)
        return [c for group in [char * common[char] for char in common] for c in group]


# ============================================================================

# 1003. Check If Word Is Valid After Substitutions
# Difficulty: Medium
# link: https://leetcode.com/problems/check-if-word-is-valid-after-substitutions/
# Companies: Nutanix
# Categories: String,Stack

# ----------------------------------------------------------------------------

class Solution(object):

    def isValid(self, S):
        stack = []
        for char in S:
            if char == "c":
                if len(stack) < 2 or stack.pop() != 'b' or stack.pop() != 'a': return False
            else: stack.append(char)
        return not stack

    # brute force
    def _isValid(self, S):
        prev, cur = "", S
        while prev != cur:
            prev = cur
            cur = cur.replace("abc", "")
        return cur == ""


# ============================================================================

# 1004. Max Consecutive Ones III
# Difficulty: Medium
# link: https://leetcode.com/problems/max-consecutive-ones-iii/
# Companies: Facebook
# Categories: Two Pointers,Sliding Window

# ----------------------------------------------------------------------------

class Solution(object):
    def longestOnes(self, A, K):
        res = start = 0
        allowable_zeros = K
        for end, bit in enumerate(A):
            if not bit: allowable_zeros -= 1
            while allowable_zeros < 0:
                allowable_zeros += bool(A[start] == 0)
                start += 1
            res = max(res, end - start + 1)
        return res


# ============================================================================

# 1005. Maximize Sum Of Array After K Negations
# Difficulty: Easy
# link: https://leetcode.com/problems/maximize-sum-of-array-after-k-negations/
# Companies: druva
# Categories: Greedy

# ----------------------------------------------------------------------------

class Solution(object):
    def largestSumAfterKNegations(self, A, K):
        while K:
            neg_idx = A.index(min(A))
            A[neg_idx] = -A[neg_idx]
            K -= 1
        return sum(A)


# ============================================================================

# 1006. Clumsy Factorial
# Difficulty: Medium
# link: https://leetcode.com/problems/clumsy-factorial/
# Companies: Microsoft
# Categories: Math

# ----------------------------------------------------------------------------

class Solution(object):
    def clumsy(self, N):
        """
        :type N: int
        :rtype: int
        """
        res = 0
        pri = N
        op_idx = 0
        for num in range(N - 1, 0,  - 1):
            op = op_idx % 4
            if op == 0:
                pri *= num
            elif op == 1:
                res += abs(pri) / num * (1 if pri > 0 else -1)
                pri = 0
            elif op == 2:
                res += num
            elif op == 3:
                pri = -num
            op_idx += 1
        return res + pri


# ============================================================================

# 1007. Minimum Domino Rotations For Equal Row
# Difficulty: Medium
# link: https://leetcode.com/problems/minimum-domino-rotations-for-equal-row/
# Companies: Amazon,Google
# Categories: Array,Greedy

# ----------------------------------------------------------------------------

class Solution(object):
    def minDominoRotations(self, A, B):
        from collections import Counter
        A_counts, B_counts, valid_nums = Counter(A), Counter(B), set(range(1, 7))
        for i in range(len(A)): valid_nums = valid_nums & set([A[i], B[i]])
        return min((min(A_counts[num], B_counts[num]) - (sum([A_counts[num], B_counts[num]]) - len(A))) for num in valid_nums) if valid_nums else -1


# ============================================================================

# 1008. Construct Binary Search Tree from Preorder Traversal
# Difficulty: Medium
# link: https://leetcode.com/problems/construct-binary-search-tree-from-preorder-traversal/
# Companies: Facebook
# Categories: Tree

# ----------------------------------------------------------------------------

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def bstFromPreorder(self, preorder):
        def bstFromPreorder(s, e):
            if s < e:
                root = TreeNode(preorder[s])
                m = s + 1
                while m < e and preorder[m] < preorder[s]: m += 1
                root.left = bstFromPreorder(s + 1, m)
                root.right = bstFromPreorder(m, e)
                return root
        return bstFromPreorder(0, len(preorder))


# ============================================================================

# 1018. Binary Prefix Divisible By 5
# Difficulty: Easy
# link: https://leetcode.com/problems/binary-prefix-divisible-by-5/
# Companies:
# Categories: Array

# ----------------------------------------------------------------------------

class Solution(object):
    def prefixesDivBy5(self, A):
        """
        :type A: List[int]
        :rtype: List[bool]
        """
        res = []
        num = 0
        for bit in A:
            num = (num << 1) | bit
            res.append(num % 5 == 0)
        return res


# ============================================================================

# 1019. Next Greater Node In Linked List
# Difficulty: Medium
# link: https://leetcode.com/problems/next-greater-node-in-linked-list/
# Companies: Uber,Amazon
# Categories: Linked List,Stack

# ----------------------------------------------------------------------------

# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def nextLargerNodes(self, head):
        """
        :type head: ListNode
        :rtype: List[int]
        """
        cur = head
        lst, res, stack = [[] for _ in range(3)]
        while cur:
            lst.append(cur.val)
            cur = cur.next
        for num in reversed(lst):
            while stack and num >= stack[-1]: stack.pop()
            res.append(stack[-1] if stack else 0)
            stack.append(num)
        return reversed(res)


# ============================================================================

# 1020. Number of Enclaves
# Difficulty: Medium
# link: https://leetcode.com/problems/number-of-enclaves/
# Companies: Google
# Categories: Depth-first Search

# ----------------------------------------------------------------------------

class Solution(object):
    def numEnclaves(self, A):
        """
        :type A: List[List[int]]
        :rtype: int
        """
        if not A: return 0
        m, n = len(A), len(A[0])
        to_walk = set()
        for i in range(m): to_walk.update({(i, 0), (i, n - 1)})
        for i in range(n): to_walk.update({(0, i), (m - 1, i)})

        while to_walk:
            x, y = to_walk.pop()
            if (0 <= x < m) and (0 <= y < n) and A[x][y]:
                A[x][y] = 0
                to_walk.update({(dir_x, dir_y)
                                for dir_x, dir_y in [[x - 1, y],
                                                     [x + 1, y],
                                                     [x, y - 1],
                                                     [x, y + 1]]})

        return sum(map(sum, A))


# ============================================================================

# 1021. Remove Outermost Parentheses
# Difficulty: Easy
# link: https://leetcode.com/problems/remove-outermost-parentheses/
# Companies: Facebook
# Categories: Stack

# ----------------------------------------------------------------------------

class Solution(object):
    def removeOuterParentheses(self, S):
        """
        :type S: str
        :rtype: str
        """
        S = list(S)
        cnt = 0
        for i, char in enumerate(S):
            if char == '(':
                cnt += 1
                if cnt == 1: S[i] = ""
            else:
                cnt -= 1
                if cnt == 0: S[i] = ""
        return "".join(S)


# ============================================================================

# 1022. Sum of Root To Leaf Binary Numbers
# Difficulty: Easy
# link: https://leetcode.com/problems/sum-of-root-to-leaf-binary-numbers/
# Companies: Amazon
# Categories: Tree

# ----------------------------------------------------------------------------

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def sumRootToLeaf(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        def get_sum(cur_int, node, mod):
            if not node: return 0
            cur_int = (cur_int << 1 | node.val) % mod
            if not node.left and not node.right: return cur_int
            else: return (get_sum(cur_int, node.left, mod) + get_sum(cur_int, node.right, mod)) % mod
        return get_sum(0, root, (10**9 + 7))


# ============================================================================

# 1023. Camelcase Matching
# Difficulty: Medium
# link: https://leetcode.com/problems/camelcase-matching/
# Companies: Amazon
# Categories: String,Trie

# ----------------------------------------------------------------------------

class Solution(object):
    def camelMatch(self, queries, pattern):
        """
        :type queries: List[str]
        :type pattern: str
        :rtype: List[bool]
        """
        low = "[a-z]*"
        pat = re.compile("^%s%s%s$" %(low, low.join(list(pattern)), low))
        return [pat.match(word) for word in queries]


# ============================================================================

# 1024. Video Stitching
# Difficulty: Medium
# link: https://leetcode.com/problems/video-stitching/
# Companies: Amazon
# Categories: Dynamic Programming

# ----------------------------------------------------------------------------

class Solution(object):
    def videoStitching(self, clips, T):
        clips = sorted(clips, reverse=True)
        cur_end = res = new_end = 0
        while clips and cur_end < T and cur_end >= clips[-1][0]:
            while clips and clips[-1][0] <= cur_end:
                s, e = clips.pop()
                new_end = max(new_end, e)
            res += 1
            cur_end = new_end
        return res if new_end >= T else -1


# ============================================================================

# 1026. Maximum Difference Between Node and Ancestor
# Difficulty: Medium
# link: https://leetcode.com/problems/maximum-difference-between-node-and-ancestor/
# Companies: Amazon
# Categories: Tree,Depth-first Search

# ----------------------------------------------------------------------------

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def maxAncestorDiff(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        def dfs(node, l=None, s=None):
            if not node: return 0
            l = max(node.val if l is None else l, node.val)
            s = min(node.val if s is None else s, node.val)
            return max(dfs(node.left, l, s), dfs(node.right, l, s), l - s)


        return dfs(root)


# ============================================================================

# 1030. Matrix Cells in Distance Order
# Difficulty: Easy
# link: https://leetcode.com/problems/matrix-cells-in-distance-order/
# Companies: Yahoo
# Categories: Sort

# ----------------------------------------------------------------------------

class Solution(object):
    def allCellsDistOrder(self, R, C, r0, c0):
        return [rc for _, rc in sorted([(abs(r - r0) + abs(c - c0), (r, c)) for r in range(R) for c in range(C)])]


# ============================================================================

# 1031. Maximum Sum of Two Non-Overlapping Subarrays
# Difficulty: Medium
# link: https://leetcode.com/problems/maximum-sum-of-two-non-overlapping-subarrays/
# Companies: Snapchat
# Categories: Array

# ----------------------------------------------------------------------------

class Solution(object):
    def maxSumTwoNoOverlap(self, A, L, M):
        """
        :type A: List[int]
        :type L: int
        :type M: int
        :rtype: int
        """
        l_sum = [0]
        for num in A: l_sum.append(l_sum[-1] + num)
        return max(l_sum[i + L] - l_sum[i] + l_sum[j + M] - l_sum[j]
                   for i in range(len(A) - L + 1)
                   for j in range(len(A) - M + 1)
                   if j >= i + L or i >= j + M)


# ============================================================================

# 1032. Stream of Characters
# Difficulty: Hard
# link: https://leetcode.com/problems/stream-of-characters/
# Companies: Amazon,Google,Facebook
# Categories: Trie

# ----------------------------------------------------------------------------

class StreamChecker(object):

    def __init__(self, words):
        """
        :type words: List[str]
        """
        self.tri_tree = {}
        for word in words:
            cur = self.tri_tree
            for char in word:
                cur = cur.setdefault(char, {})
            cur[True] = True
        self.nodes = []

    def query(self, letter):
        self.nodes.append(self.tri_tree)
        new_nodes = []
        res = False
        for node in self.nodes:
            if letter in node:
                if True in node[letter]:
                    res = True
                new_nodes.append(node[letter])
        self.nodes = new_nodes
        return res

    def __init__2(self, words):
        """
        :type words: List[str]
        """
        self.tri_tree = {}
        for word in words:
            cur = self.tri_tree
            for char in reversed(word):
                cur = cur.setdefault(char, {})
            cur[True] = True
        self.stack = []

    def query_2(self, letter):
        """
        :type letter: str
        :rtype: bool
        """
        self.stack.append(letter)
        cur = self.tri_tree
        for char in reversed(self.stack):
            if char not in cur: return False
            elif True in cur[char]: return True
            cur = cur[char]

        return False


# Your StreamChecker object will be instantiated and called as such:
# obj = StreamChecker(words)
# param_1 = obj.query(letter)


# ============================================================================

# 1037. Valid Boomerang
# Difficulty: Easy
# link: https://leetcode.com/problems/valid-boomerang/
# Companies: Google
# Categories: Math

# ----------------------------------------------------------------------------

class Solution(object):
    def isBoomerang(self, points):
        """
        :type points: List[List[int]]
        :rtype: bool
        """
        if len(set(map(tuple, points))) != 3: return False
        a, b, c = sorted(points)
        vert_cnt = len(set([p[0] for p in points]))

        if vert_cnt == 1: return False
        elif vert_cnt == 2: return True
        else: return float(b[1] - a[1]) / (b[0] - a[0])  !=  float(c[1] - b[1]) / (c[0] - b[0])
    # a[1] b[1] c[1]
    # a[0] b[0] c[0]


# ============================================================================

# 1038. Binary Search Tree to Greater Sum Tree
# Difficulty: Medium
# link: https://leetcode.com/problems/binary-search-tree-to-greater-sum-tree/
# Companies: Amazon,Apple
# Categories: Binary Search Tree

# ----------------------------------------------------------------------------

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def bstToGst(self, root):
        """
        :type root: TreeNode
        :rtype: TreeNode
        """
        lst = []
        def flatten(node):
            if node:
                flatten(node.left)
                lst.append(node)
                flatten(node.right)
        flatten(root)
        sum_so_far = 0
        for node in reversed(lst):
            node.val = sum_so_far = node.val + sum_so_far
        return root


# ============================================================================

# 1041. Robot Bounded In Circle
# Difficulty: Medium
# link: https://leetcode.com/problems/robot-bounded-in-circle/
# Companies: Twitter,Qualtrics
# Categories: Math

# ----------------------------------------------------------------------------

class Solution(object):
    def isRobotBounded(self, instructions):
        """
        :type instructions: str
        :rtype: bool
        """
        direction = x = y = 0
        for i in range(4):
            for inst in instructions:
                if inst == "G":
                    if direction % 4 == 0: y += 1
                    elif direction % 4 == 1: x += 1
                    elif direction % 4 == 2: y -= 1
                    elif direction % 4 == 3: x -= 1
                elif inst == "R": direction += 1
                elif inst == "L": direction -= 1
        return  x == y == direction % 4 == 0


# ============================================================================

# 1090. Largest Values From Labels
# Difficulty: Medium
# link: https://leetcode.com/problems/largest-values-from-labels/
# Companies: Google
# Categories: Hash Table,Greedy

# ----------------------------------------------------------------------------

class Solution(object):
    def largestValsFromLabels(self, values, labels, num_wanted, use_limit):
        from collections import Counter
        import heapq
        vals_lbls = zip([-v for v in values], labels)
        cnt = Counter()
        heapq.heapify(vals_lbls)
        res=0
        while vals_lbls and num_wanted:
            v, l = heapq.heappop(vals_lbls)
            v = -v
            cnt[l] += 1
            if cnt[l] <= use_limit:
                res += v
                num_wanted -= 1
        return res


# ============================================================================

# 1155. Number of Dice Rolls With Target Sum
# Difficulty: Medium
# link: https://leetcode.com/problems/number-of-dice-rolls-with-target-sum/
# Companies: Amazon,Google
# Categories: Dynamic Programming

# ----------------------------------------------------------------------------

class Solution(object):
    def numRollsToTarget(self, d, f, target):

        def _cnt(d=d,tar=target,memo={}):
            if (d,tar) in memo: return memo[(d,tar)]
            elif d==0 and tar==0: return 1
            elif d>0 and tar>0:
                return memo.setdefault(
                    (d,tar),
                    sum(cnt(d-1, tar-j) for j in range(1, f+1)))
            return 0

        def cnt():
            from itertools import chain
            m, n = target + 1, d + 1
            DP=[[0] * m for _ in range(n)]
            DP[0][0] = 1

            for i in range(n):
                for j in range(m):
                    if i > 0:
                        DP[i][j] = sum(chain(
                                        (DP[i-1][j-c]
                                        for c in range(1,f+1)
                                        if j-c >=0), (0,)))
            return DP[-1][-1]
        return cnt() % (10**9 + 7)
```
