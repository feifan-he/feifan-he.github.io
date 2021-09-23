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


```
