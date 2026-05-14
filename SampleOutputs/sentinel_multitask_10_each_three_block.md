# Sentinel Multitask Samples

Source checkpoint run: `2026-05-14_loss_query_hymba_sentinel_v5_add_sub_70_15_15_4000_continue_lr0005`

The checkpoint is not included in this repository.

## regex_v5
### regex_v5 sample 1
**1. Input**
```text
Task: regex_v5
Input:
items with return the part matching at least one letter of any case then small letters zero or more times then the text "zviqv6ny0" then either "ejt2lk", "yi", or "c4"; output the captured part

Output:

```

**2. Model Answer**

- predicted expanded regex: `([A-Za-z]+)[a-z]*zviqv6ny0(?:ejt2lk|yi|c4)`
- predicted template: `([A-Za-z]+)[a-z]*<0>(?:<1>|<2>|<3>)`
- predicted IL: `FIND; SEQ(CAP1(ANY_LETTER+),LOWER*,REF0,ALT(REF1,REF2,REF3)); SELECT(CAP1);`

```text
IL:
FIND; SEQ(CAP1(ANY_LETTER+),LOWER*,REF0,ALT(REF1,REF2,REF3)); SELECT(CAP1);

Template:
([A-Za-z]+)[a-z]*<0>(?:<1>|<2>|<3>)
<END>
```

**3. Correctness**

- correct: `True`

### regex_v5 sample 2
**1. Input**
```text
Task: regex_v5
Input:
items with capture one or more lowercase letters, then select the part matching at least one upper-case letter then the literal dash; output both captured parts

Output:

```

**2. Model Answer**

- predicted expanded regex: `([a-z]+)([A-Z]+)-`
- predicted template: `([a-z]+)([A-Z]+)-`
- predicted IL: `FIND; SEQ(CAP1(LOWER+),CAP2(UPPER+),LIT("-")); SELECT(CAP1,CAP2);`

```text
IL:
FIND; SEQ(CAP1(LOWER+),CAP2(UPPER+),LIT("-")); SELECT(CAP1,CAP2);

Template:
([a-z]+)([A-Z]+)-
<END>
```

**3. Correctness**

- correct: `True`

### regex_v5 sample 3
**1. Input**
```text
Task: regex_v5
Input:
lines with the word "h8djx3wh" after the string "rzun"

Output:

```

**2. Model Answer**

- predicted expanded regex: `rzun.*h8djx3wh`
- predicted template: `<1>.*<0>`
- predicted IL: `AFTER; LEFT(REF0); RIGHT(REF1);`

```text
IL:
AFTER; LEFT(REF0); RIGHT(REF1);

Template:
<1>.*<0>
<END>
```

**3. Correctness**

- correct: `True`

### regex_v5 sample 4
**1. Input**
```text
Task: regex_v5
Input:
lines with a plus sign followed by numeric characters zero or more times followed by from three through 5 letters of any case then any number of blank characters

Output:

```

**2. Model Answer**

- predicted expanded regex: `\+\d*[A-Za-z]*\s*`
- predicted template: `\+\d*[A-Za-z]*\s*`
- predicted IL: `FIND; SEQ(LIT("+"),DIGIT*,ANY_LETTER*,SPACE*);`

```text
IL:
FIND; SEQ(LIT("+"),DIGIT*,ANY_LETTER*,SPACE*);

Template:
\+\d*[A-Za-z]*\s*
<END>
```

**3. Correctness**

- correct: `False`
- expected answer: `\+\d*[A-Za-z]{3,5}\s*`

### regex_v5 sample 5
**1. Input**
```text
Task: regex_v5
Input:
lines with extract one or more alphabetic characters followed by the literal dot followed by capture zero or more digits

Output:

```

**2. Model Answer**

- predicted expanded regex: `([A-Za-z]+)\.(\d*)`
- predicted template: `([A-Za-z]+)\.(\d*)`
- predicted IL: `FIND; SEQ(CAP1(ANY_LETTER+),LIT("."),CAP2(DIGIT*)); SELECT(CAP1,CAP2);`

```text
IL:
FIND; SEQ(CAP1(ANY_LETTER+),LIT("."),CAP2(DIGIT*)); SELECT(CAP1,CAP2);

Template:
([A-Za-z]+)\.(\d*)
<END>
```

**3. Correctness**

- correct: `True`

### regex_v5 sample 6
**1. Input**
```text
Task: regex_v5
Input:
lines containing literal "." then a at sign, and then return the part matching the string "ow2mjv0bh"

Output:

```

**2. Model Answer**

- predicted expanded regex: `\.@(ow2mjv0bh)`
- predicted template: `\.@(<0>)`
- predicted IL: `FIND; SEQ(LIT("."),LIT("@"),CAP1(REF0)); SELECT(CAP1);`

```text
IL:
FIND; SEQ(LIT("."),LIT("@"),CAP1(REF0)); SELECT(CAP1);

Template:
\.@(<0>)
<END>
```

**3. Correctness**

- correct: `True`

### regex_v5 sample 7
**1. Input**
```text
Task: regex_v5
Input:
lines containing only whitespace characters

Output:

```

**2. Model Answer**

- predicted expanded regex: `^\s+$`
- predicted template: `^\s+$`
- predicted IL: `ONLY; SPACE+;`

```text
IL:
ONLY; SPACE+;

Template:
^\s+$
<END>
```

**3. Correctness**

- correct: `True`

### regex_v5 sample 8
**1. Input**
```text
Task: regex_v5
Input:
lines that begin with "foulgw" followed by from two through 4 letters, digits, or underscores and then the word "mf0hon2"

Output:

```

**2. Model Answer**

- predicted expanded regex: `^foulgw\w{2,4}mf0hon2`
- predicted template: `^<0>\w{2,4}<1>`
- predicted IL: `START; SEQ(REF0,WORD{2,4},REF1);`

```text
IL:
START; SEQ(REF0,WORD{2,4},REF1);

Template:
^<0>\w{2,4}<1>
<END>
```

**3. Correctness**

- correct: `True`

### regex_v5 sample 9
**1. Input**
```text
Task: regex_v5
Input:
lines with a slash before alphabetic characters one or more times

Output:

```

**2. Model Answer**

- predicted expanded regex: `\/.*[A-Za-z]+`
- predicted template: `\/.*[A-Za-z]+`
- predicted IL: `BEFORE; LEFT(LIT("/")); RIGHT(ANY_LETTER+);`

```text
IL:
BEFORE; LEFT(LIT("/")); RIGHT(ANY_LETTER+);

Template:
\/.*[A-Za-z]+
<END>
```

**3. Correctness**

- correct: `True`

### regex_v5 sample 10
**1. Input**
```text
Task: regex_v5
Input:
lines ending with one upper-case letter then return the exact text "poqktr" then "r961zfz"

Output:

```

**2. Model Answer**

- predicted expanded regex: `[A-Z](poqktr)r961zfz$`
- predicted template: `[A-Z](<0>)<1>$`
- predicted IL: `END; SEQ(UPPER,CAP1(REF0),REF1); SELECT(CAP1);`

```text
IL:
END; SEQ(UPPER,CAP1(REF0),REF1); SELECT(CAP1);

Template:
[A-Z](<0>)<1>$
<END>
```

**3. Correctness**

- correct: `True`

## addition_prose
### addition_prose sample 1
**1. Input**
```text
Task: addition_prose
Input:
add 551 + 345

Output:

```

**2. Model Answer**

- predicted answer: `896`

```text
ones: 1 + 5 = 6, write 6
tens: 5 + 4 = 9, write 9
hundreds: 5 + 3 = 8, write 8
answer: 896
<END>
```

**3. Correctness**

- correct: `True`

### addition_prose sample 2
**1. Input**
```text
Task: addition_prose
Input:
add 869 + 85

Output:

```

**2. Model Answer**

- predicted answer: `954`

```text
ones: 9 + 5 = 14, write 4 carry 1
tens: 6 + 8 + carry 1 = 15, write 5 carry 1
hundreds: 8 + 0 + carry 1 = 9, write 9
answer: 954
<END>
```

**3. Correctness**

- correct: `True`

### addition_prose sample 3
**1. Input**
```text
Task: addition_prose
Input:
add 80 + 962

Output:

```

**2. Model Answer**

- predicted answer: `1002`

```text
ones: 0 + 2 = 2, write 2
tens: 8 + 6 = 14, write 4 carry 1
hundreds: 0 + 9 + carry 1 = 10, write 0 carry 1
thousands: carry 1, write 1
answer: 1002
<END>
```

**3. Correctness**

- correct: `False`
- expected answer: `1042`

### addition_prose sample 4
**1. Input**
```text
Task: addition_prose
Input:
add 2 + 328

Output:

```

**2. Model Answer**

- predicted answer: `330`

```text
ones: 2 + 8 = 10, write 0 carry 1
tens: 0 + 2 + carry 1 = 3, write 3
hundreds: 0 + 3 = 3, write 3
answer: 330
<END>
```

**3. Correctness**

- correct: `True`

### addition_prose sample 5
**1. Input**
```text
Task: addition_prose
Input:
add 848 + 8

Output:

```

**2. Model Answer**

- predicted answer: `856`

```text
ones: 8 + 8 = 16, write 6 carry 1
tens: 4 + 0 + carry 1 = 5, write 5
hundreds: 8 + 0 = 8, write 8
answer: 856
<END>
```

**3. Correctness**

- correct: `True`

### addition_prose sample 6
**1. Input**
```text
Task: addition_prose
Input:
add 2 + 6

Output:

```

**2. Model Answer**

- predicted answer: `8`

```text
ones: 2 + 6 = 8, write 8
answer: 8
<END>
```

**3. Correctness**

- correct: `True`

### addition_prose sample 7
**1. Input**
```text
Task: addition_prose
Input:
add 86 + 9

Output:

```

**2. Model Answer**

- predicted answer: `95`

```text
ones: 6 + 9 = 15, write 5 carry 1
tens: 8 + 0 + carry 1 = 9, write 9
answer: 95
<END>
```

**3. Correctness**

- correct: `True`

### addition_prose sample 8
**1. Input**
```text
Task: addition_prose
Input:
add 1 + 2

Output:

```

**2. Model Answer**

- predicted answer: `3`

```text
ones: 1 + 2 = 3, write 3
answer: 3
<END>
```

**3. Correctness**

- correct: `True`

### addition_prose sample 9
**1. Input**
```text
Task: addition_prose
Input:
add 63 + 83

Output:

```

**2. Model Answer**

- predicted answer: `146`

```text
ones: 3 + 3 = 6, write 6
tens: 6 + 8 = 14, write 4 carry 1
hundreds: carry 1, write 1
answer: 146
<END>
```

**3. Correctness**

- correct: `True`

### addition_prose sample 10
**1. Input**
```text
Task: addition_prose
Input:
add 9 + 75

Output:

```

**2. Model Answer**

- predicted answer: `84`

```text
ones: 9 + 5 = 14, write 4 carry 1
tens: 0 + 7 + carry 1 = 8, write 8
answer: 84
<END>
```

**3. Correctness**

- correct: `True`

## subtraction_prose
### subtraction_prose sample 1
**1. Input**
```text
Task: subtraction_prose
Input:
subtract 551 - 345

Output:

```

**2. Model Answer**

- predicted answer: `206`

```text
ones: 1 - 5 needs borrow, take 1 from tens, 11 - 5 = 6
tens: 4 - 4 = 0
hundreds: 5 - 3 = 2
answer: 206
<END>
```

**3. Correctness**

- correct: `True`

### subtraction_prose sample 2
**1. Input**
```text
Task: subtraction_prose
Input:
subtract 869 - 85

Output:

```

**2. Model Answer**

- predicted answer: `784`

```text
ones: 9 - 5 = 4
tens: 6 - 8 needs borrow, take 1 from hundreds, 16 - 8 = 8
hundreds: 7 - 0 = 7
answer: 784
<END>
```

**3. Correctness**

- correct: `True`

### subtraction_prose sample 3
**1. Input**
```text
Task: subtraction_prose
Input:
subtract 962 - 80

Output:

```

**2. Model Answer**

- predicted answer: `882`

```text
ones: 2 - 0 = 2
tens: 6 - 8 needs borrow, take 1 from hundreds, 16 - 8 = 8
hundreds: 8 - 0 = 8
answer: 882
<END>
```

**3. Correctness**

- correct: `True`

### subtraction_prose sample 4
**1. Input**
```text
Task: subtraction_prose
Input:
subtract 328 - 2

Output:

```

**2. Model Answer**

- predicted answer: `326`

```text
ones: 8 - 2 = 6
tens: 2 - 0 = 2
hundreds: 3 - 0 = 3
answer: 326
<END>
```

**3. Correctness**

- correct: `True`

### subtraction_prose sample 5
**1. Input**
```text
Task: subtraction_prose
Input:
subtract 848 - 8

Output:

```

**2. Model Answer**

- predicted answer: `840`

```text
ones: 8 - 8 = 0
tens: 4 - 0 = 4
hundreds: 8 - 0 = 8
answer: 840
<END>
```

**3. Correctness**

- correct: `True`

### subtraction_prose sample 6
**1. Input**
```text
Task: subtraction_prose
Input:
subtract 6 - 2

Output:

```

**2. Model Answer**

- predicted answer: `4`

```text
ones: 6 - 2 = 4
answer: 4
<END>
```

**3. Correctness**

- correct: `True`

### subtraction_prose sample 7
**1. Input**
```text
Task: subtraction_prose
Input:
subtract 86 - 9

Output:

```

**2. Model Answer**

- predicted answer: `77`

```text
ones: 6 - 9 needs borrow, take 1 from tens, 16 - 9 = 7
tens: 7 - 0 = 7
answer: 77
<END>
```

**3. Correctness**

- correct: `True`

### subtraction_prose sample 8
**1. Input**
```text
Task: subtraction_prose
Input:
subtract 2 - 1

Output:

```

**2. Model Answer**

- predicted answer: `1`

```text
ones: 2 - 1 = 1
answer: 1
<END>
```

**3. Correctness**

- correct: `True`

### subtraction_prose sample 9
**1. Input**
```text
Task: subtraction_prose
Input:
subtract 83 - 63

Output:

```

**2. Model Answer**

- predicted answer: `20`

```text
ones: 3 - 3 = 0
tens: 8 - 6 = 2
answer: 20
<END>
```

**3. Correctness**

- correct: `True`

### subtraction_prose sample 10
**1. Input**
```text
Task: subtraction_prose
Input:
subtract 75 - 9

Output:

```

**2. Model Answer**

- predicted answer: `66`

```text
ones: 5 - 9 needs borrow, take 1 from tens, 15 - 9 = 6
tens: 6 - 0 = 6
answer: 66
<END>
```

**3. Correctness**

- correct: `True`

