/* Smoke test for jade-protocol.so — validates the C ABI links and the callback
 * surface compiles. Run with a live daemon to see real tool-call streaming.
 *
 * Build (from repo root, after `cargo build -p jade-protocol-ffi`):
 *   cc -I jade-protocol-ffi/include jade-protocol-ffi/examples/smoke.c \
 *      -L target/debug -ljade_protocol -o /tmp/jade_smoke
 *   /tmp/jade_smoke
 */
#include <stdio.h>
#include <stddef.h>
#include "jade_protocol.h"

static void on_meta(const char *model, void *ud) {
    (void)ud;
    printf("[meta] model=%s\n", model);
}
static void on_token(const char *text, uintptr_t len, void *ud) {
    (void)ud;
    printf("%.*s", (int)len, text);
    fflush(stdout);
}
static void on_tool_call(const char *name, const char *args, void *ud) {
    (void)ud;
    printf("\n[tool_call] name=%s args=%s\n", name, args);
}
static void on_done(uint64_t tokens, void *ud) {
    (void)ud;
    printf("\n[done] tokens=%llu\n", (unsigned long long)tokens);
}
static void on_error(const char *msg, void *ud) {
    (void)ud;
    printf("\n[error] %s\n", msg);
}

int main(void) {
    JadeConn *c = jade_connect(NULL); /* $HOME/.jade/llm.sock */
    if (!c) {
        printf("could not connect to daemon (expected if jaded is not running)\n");
        return 0;
    }

    JadeRequest *r = jade_request_messages();
    jade_request_add_msg(r, "user", "What's the weather in San Francisco?");
    jade_request_enable_tools(r, "root ::= \"{\" [^}]* \"}\"");
    jade_request_set_max_tokens(r, 128);

    JadeCallbacks cb = {
        .on_meta = on_meta,
        .on_token = on_token,
        .on_tool_call = on_tool_call,
        .on_done = on_done,
        .on_error = on_error,
    };

    int rc = jade_infer(c, r, &cb, NULL);
    printf("\njade_infer rc=%d\n", rc);

    jade_request_free(r);
    jade_disconnect(c);
    return rc == JADE_OK ? 0 : 1;
}
