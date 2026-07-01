---
name: Bug report
about: Create a report to help us improve.
title: "[BUG]"
labels: bug
assignees: Alexhuszagh

---

## Description

Please include a clear and concise description of the bug. If the bug includes a security vulnerability, please privately report the issue to the [maintainer](mailto:ahuszagh@gmail.com).

## Test case

Please provide a short, complete (with crate import, etc) test case for
the issue, showing clearly the expected and obtained results.

Example test case:

```rust
use stackvector::StackVec;

fn main() {
  let buf = [1, 2, 3, 4, 5];
  let stack_vec: StackVec<i32, 5> = StackVec::from_buf(buf);
  assert_eq!(&*stack_vec, &[1, 2, 3, 4, 5]);
}
```

## Additional Context

Add any other context about the problem here.
