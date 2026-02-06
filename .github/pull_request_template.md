<!--
(You can either delete or dance around these HTML comments -- they're only here to provide instruction)
-->

<!--
Please use this template when creating new Pull Requests in this repo. This allows for the team to have a consistent means of defining and tracking work being done to this codebase.

Don't forget to add relevant lables before creating!
-->

## Jira Story Link
[HTMRL-###](https://psu-capstone.atlassian.net/browse/HTMRL-###)

## Summary

<!--
Provide a brief summary of what this PR is designed to address (the "why" more than the "what"). Yes, this should be captured in the Jira link above but the more places we have details and logic flow, the better. If applicable, also provide reproducibility steps, important context, etc.

Example:

Currently, errors in the SDR encoder are not wrapped. Raw errors are sent back up the callstack. This is not ideal since it does provide a callstack for us and makes it more difficult to determine where in the logic the error is taking place. This PR updates the logic in the SDR encoder method to better wrap errors before returning them up the callstack
-->

# Dependencies
<!--
If applicable, link to any other PRs this PR depends upon and why.

Example:

- [HTMRL-###](https://psu-capstone.atlassian.net/browse/HTMRL-###): We need to wait on this PR so the basic implementation for error wrapping is in place first. 
-->

## Testing performed

### Summary

<!--
Provide a high-level summary of test results from the procedures outlined below, if applicable
-->

### How was this tested

<!--
Include a detailed description of testing steps performed
-->

## Screenshots

<!--
If applicable, supply meaningful screenshots to assist the reviewer. For instance, if the reviewer is expecting a certain output, give screenshots showing that output so that they can try and reproduce it
-->

## Tech debt this creates

<!--
NOTE: If you need to put something here, you must create a story or task in JIRA that captures the work needed to be done to resolve this tech debt so that we do not lose track of it

If applicable, a description of any tech debt created as a result of this PR's merge.

Example:
These changes ended up with instances where error wrapping is not being done in a consistent way across all calls in the repo. Ideally we want everything to be consistent

This is captured in https://psu-capstone.atlassian.net/browse/HTMRL-###
-->


## Reviewer to check:

***Author to leave this section & bullets for reviewer***
- [ ] Functional test was ran as part of testing notes
- [ ] Code formatting consistency
- [ ] Presence of docstring comments, in-line comments, logic explanations where needed
- [ ] Log statements are added anywhere as appropriate
- [ ] If unit-testable, presence of meaningful unit tests
