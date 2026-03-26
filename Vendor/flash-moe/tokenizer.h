/*
 * tokenizer.h — Single-header C BPE tokenizer for Qwen3/GPT-style models
 *
 * Usage:
 *   #define TOKENIZER_IMPL   // in exactly one .c file
 *   #include "tokenizer.h"
 *
 *   bpe_tokenizer tok;
 *   bpe_load(&tok, "tokenizer.bin");
 *   uint32_t ids[4096];
 *   int n = bpe_encode(&tok, "Hello, world!", ids, 4096);
 *   bpe_free(&tok);
 */
#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <stdint.h>
#include <stdbool.h>

#define BPE_MAX_TOKEN_LEN 256
#define BPE_MAX_PIECES 8192

typedef struct {
    char    *str;       // UTF-8 string (owned, null-terminated)
    uint16_t len;       // byte length
    uint32_t id;        // token ID
} bpe_vocab_entry;

typedef struct {
    char    *a, *b;     // merge pair strings (owned)
    uint16_t len_a, len_b;
} bpe_merge;

typedef struct {
    char    *str;       // e.g. "<|im_start|>"
    uint16_t len;
    uint32_t id;
} bpe_added_token;

typedef struct {
    bpe_vocab_entry *vocab;
    uint32_t         vocab_size;
    bpe_merge       *merges;
    uint32_t         num_merges;
    bpe_added_token *added;
    uint32_t         num_added;
    uint32_t ht_mask, *ht_ids; char **ht_keys; uint16_t *ht_klens;  // vocab hash
    uint32_t mt_mask, *mt_prio; char **mt_keys; uint16_t *mt_klens; // merge hash
    uint32_t byte_char[256]; // GPT-2 byte-to-unicode
    uint8_t  char_byte[512]; // reverse mapping
} bpe_tokenizer;

int  bpe_load(bpe_tokenizer *tok, const char *path);
int  bpe_encode(const bpe_tokenizer *tok, const char *text, uint32_t *out_ids, int max_ids);
void bpe_free(bpe_tokenizer *tok);

#endif // TOKENIZER_H

// ============================================================================
// Implementation
// ============================================================================
#ifdef TOKENIZER_IMPL

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

static uint32_t bpe_hash(const char *s, uint16_t len) {
    uint32_t h = 2166136261u;
    for (uint16_t i = 0; i < len; i++) {
        h ^= (uint8_t)s[i];
        h *= 16777619u;
    }
    return h;
}

static void ht_insert(uint32_t *ids, char **keys, uint16_t *klens,
                       uint32_t mask, const char *key, uint16_t klen, uint32_t id) {
    uint32_t h = bpe_hash(key, klen) & mask;
    while (ids[h] != 0xFFFFFFFF) h = (h + 1) & mask;
    ids[h] = id;
    keys[h] = (char*)key;
    klens[h] = klen;
}

static uint32_t ht_lookup(const uint32_t *ids, char *const *keys, const uint16_t *klens,
                           uint32_t mask, const char *key, uint16_t klen) {
    uint32_t h = bpe_hash(key, klen) & mask;
    while (ids[h] != 0xFFFFFFFF) {
        if (klens[h] == klen && memcmp(keys[h], key, klen) == 0)
            return ids[h];
        h = (h + 1) & mask;
    }
    return 0xFFFFFFFF;
}

static void build_byte_unicode_table(bpe_tokenizer *tok) {
    int n = 0;
    for (int b = 0; b < 256; b++) {
        if ((b >= 0x21 && b <= 0x7E) || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF))
            tok->byte_char[b] = (uint32_t)b;
        else { tok->byte_char[b] = 256 + n; n++; }
    }
    memset(tok->char_byte, 0, sizeof(tok->char_byte));
    for (int b = 0; b < 256; b++) {
        uint32_t cp = tok->byte_char[b]; if (cp < 512) tok->char_byte[cp] = (uint8_t)b;
    }
}

static int bytes_to_bpe_str(const bpe_tokenizer *tok, const uint8_t *raw, int raw_len,
                             char *out, int out_cap) {
    int pos = 0;
    for (int i = 0; i < raw_len && pos < out_cap - 4; i++) {
        uint32_t cp = tok->byte_char[raw[i]];
        if (cp < 0x80) out[pos++] = (char)cp;
        else if (cp < 0x800) { out[pos++] = 0xC0|(cp>>6); out[pos++] = 0x80|(cp&0x3F); }
        else { out[pos++] = 0xE0|(cp>>12); out[pos++] = 0x80|((cp>>6)&0x3F); out[pos++] = 0x80|(cp&0x3F); }
    }
    out[pos] = '\0'; return pos;
}

static int read_u32(FILE *f, uint32_t *v) { return fread(v, 4, 1, f) == 1 ? 0 : -1; }
static int read_u16(FILE *f, uint16_t *v) { return fread(v, 2, 1, f) == 1 ? 0 : -1; }
static uint32_t next_pow2(uint32_t n) {
    n--;
    n |= n >> 1; n |= n >> 2; n |= n >> 4; n |= n >> 8; n |= n >> 16;
    return n + 1;
}

int bpe_load(bpe_tokenizer *tok, const char *path) {
    memset(tok, 0, sizeof(*tok));
    build_byte_unicode_table(tok);

    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "bpe_load: cannot open %s\n", path); return -1; }

    char magic[4];
    uint32_t version;
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "BPET", 4) != 0) goto fail;
    if (read_u32(f, &version) || version != 1) goto fail;
    if (read_u32(f, &tok->vocab_size)) goto fail;
    if (read_u32(f, &tok->num_merges)) goto fail;
    if (read_u32(f, &tok->num_added)) goto fail;

    tok->vocab = calloc(tok->vocab_size, sizeof(bpe_vocab_entry));
    for (uint32_t i = 0; i < tok->vocab_size; i++) {
        if (read_u32(f, &tok->vocab[i].id)) goto fail;
        if (read_u16(f, &tok->vocab[i].len)) goto fail;
        tok->vocab[i].str = malloc(tok->vocab[i].len + 1);
        if (fread(tok->vocab[i].str, 1, tok->vocab[i].len, f) != tok->vocab[i].len) goto fail;
        tok->vocab[i].str[tok->vocab[i].len] = '\0';
    }

    tok->merges = calloc(tok->num_merges, sizeof(bpe_merge));
    for (uint32_t i = 0; i < tok->num_merges; i++) {
        if (read_u16(f, &tok->merges[i].len_a)) goto fail;
        tok->merges[i].a = malloc(tok->merges[i].len_a + 1);
        if (fread(tok->merges[i].a, 1, tok->merges[i].len_a, f) != tok->merges[i].len_a) goto fail;
        tok->merges[i].a[tok->merges[i].len_a] = '\0';
        if (read_u16(f, &tok->merges[i].len_b)) goto fail;
        tok->merges[i].b = malloc(tok->merges[i].len_b + 1);
        if (fread(tok->merges[i].b, 1, tok->merges[i].len_b, f) != tok->merges[i].len_b) goto fail;
        tok->merges[i].b[tok->merges[i].len_b] = '\0';
    }

    tok->added = calloc(tok->num_added, sizeof(bpe_added_token));
    for (uint32_t i = 0; i < tok->num_added; i++) {
        if (read_u32(f, &tok->added[i].id)) goto fail;
        if (read_u16(f, &tok->added[i].len)) goto fail;
        tok->added[i].str = malloc(tok->added[i].len + 1);
        if (fread(tok->added[i].str, 1, tok->added[i].len, f) != tok->added[i].len) goto fail;
        tok->added[i].str[tok->added[i].len] = '\0';
    }
    fclose(f);

    uint32_t ht_size = next_pow2(tok->vocab_size * 2);
    tok->ht_mask = ht_size - 1;
    tok->ht_ids   = malloc(ht_size * sizeof(uint32_t));
    tok->ht_keys  = calloc(ht_size, sizeof(char*));
    tok->ht_klens = calloc(ht_size, sizeof(uint16_t));
    memset(tok->ht_ids, 0xFF, ht_size * sizeof(uint32_t));
    for (uint32_t i = 0; i < tok->vocab_size; i++) {
        ht_insert(tok->ht_ids, tok->ht_keys, tok->ht_klens, tok->ht_mask,
                  tok->vocab[i].str, tok->vocab[i].len, tok->vocab[i].id);
    }
    // Build merge hash table (key = "a\xFFb" -> priority index)
    uint32_t mt_size = next_pow2(tok->num_merges * 2);
    tok->mt_mask = mt_size - 1;
    tok->mt_prio  = malloc(mt_size * sizeof(uint32_t));
    tok->mt_keys  = calloc(mt_size, sizeof(char*));
    tok->mt_klens = calloc(mt_size, sizeof(uint16_t));
    memset(tok->mt_prio, 0xFF, mt_size * sizeof(uint32_t));
    for (uint32_t i = 0; i < tok->num_merges; i++) {
        uint16_t klen = tok->merges[i].len_a + 1 + tok->merges[i].len_b;
        char *key = malloc(klen);
        memcpy(key, tok->merges[i].a, tok->merges[i].len_a);
        key[tok->merges[i].len_a] = '\xff';
        memcpy(key + tok->merges[i].len_a + 1, tok->merges[i].b, tok->merges[i].len_b);
        ht_insert(tok->mt_prio, tok->mt_keys, tok->mt_klens, tok->mt_mask,
                  key, klen, i);
    }

    fprintf(stderr, "bpe_load: %u vocab, %u merges, %u added tokens\n",
            tok->vocab_size, tok->num_merges, tok->num_added);
    return 0;

fail:
    fclose(f);
    fprintf(stderr, "bpe_load: parse error in %s\n", path);
    return -1;
}

void bpe_free(bpe_tokenizer *tok) {
    for (uint32_t i = 0; i < tok->vocab_size; i++) free(tok->vocab[i].str);
    free(tok->vocab);
    for (uint32_t i = 0; i < tok->num_merges; i++) { free(tok->merges[i].a); free(tok->merges[i].b); }
    free(tok->merges);
    for (uint32_t i = 0; i < tok->num_added; i++) free(tok->added[i].str);
    free(tok->added);
    free(tok->ht_ids); free(tok->ht_keys); free(tok->ht_klens);
    uint32_t mt_cap = tok->mt_mask + 1;
    for (uint32_t i = 0; i < mt_cap; i++)
        if (tok->mt_prio[i] != 0xFFFFFFFF) free(tok->mt_keys[i]);
    free(tok->mt_prio); free(tok->mt_keys); free(tok->mt_klens);
    memset(tok, 0, sizeof(*tok));
}

typedef struct { int start, end; } bpe_span;
#define IS_WS(c) ((c)==' '||(c)=='\t'||(c)=='\n'||(c)=='\r')
#define IS_NL(c) ((c)=='\n'||(c)=='\r')
#define IS_ALPHA(c) (((c)>='A'&&(c)<='Z')||((c)>='a'&&(c)<='z'))
#define IS_ALNUM_WS(c) (IS_ALPHA(c)||((c)>='0'&&(c)<='9')||IS_WS(c)||(c)>=0xC0)

static int pretokenize(const char *text, int text_len, bpe_span *spans, int max_spans) {
    int n = 0, i = 0;
    while (i < text_len && n < max_spans) {
        uint8_t c = (uint8_t)text[i];
        if (IS_WS(c)) {
            int start = i; bool has_nl = false; int j = i;
            while (j < text_len && IS_WS((uint8_t)text[j])) {
                if (IS_NL((uint8_t)text[j])) has_nl = true; j++;
            }
            if (has_nl || j >= text_len) { spans[n++] = (bpe_span){start, j}; i = j; continue; }
            if (j - start > 1) { spans[n++] = (bpe_span){start, j-1}; i = j-1; continue; }
        }
        bool lead_sp = (c == ' ' && i+1 < text_len);
        int ws = i, wi = lead_sp ? i+1 : i;
        if (wi < text_len) {
            uint8_t wc = (uint8_t)text[wi];
            if (!lead_sp && wc == '\'' && wi+1 < text_len) {
                char nc = text[wi+1] | 0x20;
                if (nc=='s'||nc=='t'||nc=='m'||nc=='d') { spans[n++]=(bpe_span){wi,wi+2}; i=wi+2; continue; }
                if (wi+2 < text_len) { char nc2 = text[wi+2]|0x20;
                    if ((nc=='r'&&nc2=='e')||(nc=='v'&&nc2=='e')||(nc=='l'&&nc2=='l'))
                    { spans[n++]=(bpe_span){wi,wi+3}; i=wi+3; continue; }
                }
            }
            if (wc >= 0xC0 || IS_ALPHA(wc)) {
                int j = wi;
                while (j < text_len) { uint8_t jc = (uint8_t)text[j];
                    if (jc >= 0xC0) j += (jc<0xE0)?2:(jc<0xF0)?3:4;
                    else if (IS_ALPHA(jc)) j++; else break;
                }
                if (j > wi) { spans[n++] = (bpe_span){ws, j}; i = j; continue; }
            }
            if (wc>='0' && wc<='9') { spans[n++]=(bpe_span){ws,wi+1}; i=wi+1; continue; }
            if (!IS_ALNUM_WS(wc)) {
                int j = wi;
                while (j < text_len && !IS_ALNUM_WS((uint8_t)text[j])) j++;
                while (j < text_len && IS_NL((uint8_t)text[j])) j++;
                spans[n++] = (bpe_span){ws, j}; i = j; continue;
            }
        }
        spans[n++] = (bpe_span){i, i+1}; i++;
    }
    return n;
}
#undef IS_WS
#undef IS_NL
#undef IS_ALPHA
#undef IS_ALNUM_WS

typedef struct { char *str; uint16_t len; int prev, next; } bpe_piece;

static uint32_t merge_lookup(const bpe_tokenizer *tok, const char *a, uint16_t la,
                              const char *b, uint16_t lb) {
    uint16_t klen = la + 1 + lb;
    char key[BPE_MAX_TOKEN_LEN * 2 + 1];
    if (klen > sizeof(key)) return 0xFFFFFFFF;
    memcpy(key, a, la);
    key[la] = '\xff';
    memcpy(key + la + 1, b, lb);
    return ht_lookup(tok->mt_prio, tok->mt_keys, tok->mt_klens, tok->mt_mask, key, klen);
}

static int bpe_process(const bpe_tokenizer *tok, const char *bpe_str, int bpe_len,
                        uint32_t *out_ids, int max_ids) {
    if (bpe_len == 0) return 0;

    bpe_piece pieces[BPE_MAX_PIECES];
    int num_pieces = 0;
    int i = 0;
    while (i < bpe_len && num_pieces < BPE_MAX_PIECES) {
        uint8_t c = (uint8_t)bpe_str[i];
        int clen;
        if (c < 0x80) clen = 1;
        else if (c < 0xE0) clen = 2;
        else if (c < 0xF0) clen = 3;
        else clen = 4;
        if (i + clen > bpe_len) clen = bpe_len - i;

        pieces[num_pieces].str = (char*)bpe_str + i;
        pieces[num_pieces].len = (uint16_t)clen;
        pieces[num_pieces].prev = num_pieces - 1;
        pieces[num_pieces].next = num_pieces + 1;
        num_pieces++;
        i += clen;
    }
    if (num_pieces == 0) return 0;
    pieces[num_pieces - 1].next = -1;

    char arena[1024 * 16];
    int arena_pos = 0;
    int active = num_pieces;

    while (active > 1) {
        uint32_t best_prio = 0xFFFFFFFF;
        int best_idx = -1;

        int ci = 0;
        while (ci != -1) {
            int ni = pieces[ci].next;
            if (ni == -1) break;
            uint32_t prio = merge_lookup(tok, pieces[ci].str, pieces[ci].len,
                                          pieces[ni].str, pieces[ni].len);
            if (prio < best_prio) {
                best_prio = prio;
                best_idx = ci;
            }
            ci = ni;
        }

        if (best_idx == -1) break;

        int ni = pieces[best_idx].next;
        uint16_t new_len = pieces[best_idx].len + pieces[ni].len;
        if (new_len > BPE_MAX_TOKEN_LEN) break;

        if (pieces[best_idx].str + pieces[best_idx].len == pieces[ni].str) {
            pieces[best_idx].len = new_len;
        } else {
            if (arena_pos + new_len > (int)sizeof(arena)) arena_pos = 0;
            memcpy(arena + arena_pos, pieces[best_idx].str, pieces[best_idx].len);
            memcpy(arena + arena_pos + pieces[best_idx].len, pieces[ni].str, pieces[ni].len);
            pieces[best_idx].str = arena + arena_pos;
            pieces[best_idx].len = new_len;
            arena_pos += new_len;
        }

        pieces[best_idx].next = pieces[ni].next;
        if (pieces[ni].next != -1) pieces[pieces[ni].next].prev = best_idx;
        active--;
    }

    int out_n = 0;
    int ci2 = 0;
    while (ci2 != -1 && out_n < max_ids) {
        uint32_t id = ht_lookup(tok->ht_ids, tok->ht_keys, tok->ht_klens, tok->ht_mask,
                                pieces[ci2].str, pieces[ci2].len);
        if (id != 0xFFFFFFFF) {
            out_ids[out_n++] = id;
        } else {
            // Fallback: encode individual bytes
            for (int j = 0; j < pieces[ci2].len && out_n < max_ids; j++) {
                char single[4];
                uint32_t cp = tok->byte_char[(uint8_t)pieces[ci2].str[j]];
                int slen = 0;
                if (cp < 0x80) { single[0] = (char)cp; slen = 1; }
                else if (cp < 0x800) {
                    single[0] = (char)(0xC0 | (cp >> 6));
                    single[1] = (char)(0x80 | (cp & 0x3F));
                    slen = 2;
                }
                uint32_t byte_id = ht_lookup(tok->ht_ids, tok->ht_keys, tok->ht_klens,
                                             tok->ht_mask, single, (uint16_t)slen);
                if (byte_id != 0xFFFFFFFF) out_ids[out_n++] = byte_id;
            }
        }
        ci2 = pieces[ci2].next;
    }
    return out_n;
}

int bpe_encode(const bpe_tokenizer *tok, const char *text, uint32_t *out_ids, int max_ids) {
    int text_len = (int)strlen(text);
    int out_n = 0;
    int pos = 0;

    while (pos < text_len && out_n < max_ids) {
        bool found_added = false;
        int best_len = 0;
        uint32_t best_id = 0;
        for (uint32_t i = 0; i < tok->num_added; i++) {
            int alen = tok->added[i].len;
            if (alen > best_len && pos + alen <= text_len &&
                memcmp(text + pos, tok->added[i].str, alen) == 0) {
                best_len = alen;
                best_id = tok->added[i].id;
                found_added = true;
            }
        }
        if (found_added) {
            out_ids[out_n++] = best_id;
            pos += best_len;
            continue;
        }

        int chunk_end = text_len;
        for (uint32_t i = 0; i < tok->num_added; i++) {
            for (int j = pos + 1; j <= text_len - tok->added[i].len; j++) {
                if (memcmp(text + j, tok->added[i].str, tok->added[i].len) == 0) {
                    if (j < chunk_end) chunk_end = j;
                    break;
                }
            }
        }

        int chunk_len = chunk_end - pos;
        bpe_span spans[BPE_MAX_PIECES];
        int num_spans = pretokenize(text + pos, chunk_len, spans, BPE_MAX_PIECES);

        char bpe_buf[BPE_MAX_TOKEN_LEN * 4];
        for (int s = 0; s < num_spans && out_n < max_ids; s++) {
            const char *piece = text + pos + spans[s].start;
            int piece_len = spans[s].end - spans[s].start;

            int bpe_len = bytes_to_bpe_str(tok, (const uint8_t*)piece, piece_len,
                                            bpe_buf, sizeof(bpe_buf));

            out_n += bpe_process(tok, bpe_buf, bpe_len,
                                 out_ids + out_n, max_ids - out_n);
        }
        pos = chunk_end;
    }
    return out_n;
}

#endif // TOKENIZER_IMPL
