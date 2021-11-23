
# LeetCode Solutions

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


```
