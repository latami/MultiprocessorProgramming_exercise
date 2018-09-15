SECTION .text

global supportSSE3

global znccWorker_sse3

global scanline_cacheBlkData_sse

global blendWorker_sse2

global blend_2x2_sse3

extern posix_memalign

extern pthread_mutex_lock

extern pthread_mutex_unlock

;---------------------
;int supportSSE3(void)
;---------------------
; Check processor support for SSE3
; Returns 1 if supported

supportSSE3:

    push    rbx         ; cpuid modifies ebx

    mov     eax,    1
    cpuid
    ; Test  1st bit of ecx that signals SSE3-support
    mov     eax,    0x00000001
    and     eax,    ecx

    pop     rbx
    ret

;----------------------------
;void *znccWorker(void *data)
;----------------------------
; params (rdi)
; Callee saved: rbx, rbp, r12-r15

znccWorker_sse3:

    push    rbx
    push    rbp
    push    r15
    push    r14
    push    r13
    push    r12
    sub     rsp,    136

    mov     [rsp],  rdi
    mov     eax,    [rdi+48]
    mov     [rsp+40],   eax     ; bx
    mov     ecx,    eax
    shr     ecx,    1
    mov     [rsp+64],   ecx     ; blkSidex
    mov     ebx,    [rdi+52]
    imul    eax,    ebx
    mov     [rsp+44],   ebx     ; by
    mov     ecx,    ebx
    shr     ecx,    1
    mov     [rsp+68],   ecx     ; blkSidey
    mov     ebx,    eax
    shl     ebx,    2
    mov     [rsp+48],   ebx     ; bx*by*4
    add     eax,    7
    shl     eax,    2
    and     eax,    0xffffffe0
    mov     [rsp+52],   eax     ; blkStride in bytes

    mov     eax,    [rdi+24]
    mov     [rsp+56],   eax     ; width
    mov     ebx,    [rdi+28]
    mov     [rsp+60],   ebx     ; height

    sub     eax,    [rsp+64]
    mov     [rsp+72],   eax     ; width-blkSidex (x-iterators end)

    mov     rax,    [rdi+88]
    mov     [rsp+8],    rax     ; *dmap1
    mov     rax,    [rdi+96]
    mov     [rsp+16],   rax     ; *dmap2
    mov     rax,    [rdi+80]
    mov     [rsp+32],   rax     ; *displacements

    ; Save MAX_FLT vector to 16 byte aligned stack-address.
    mov     eax,        0xff7fffff
    mov     [rsp+112],  eax
    mov     [rsp+116],  eax
    mov     [rsp+120],  eax
    mov     [rsp+124],  eax

.whileTop:
    mov     r11,    [rsp]
    mov     rdi,    [r11+8]     ; load ptr for mutex
    call    pthread_mutex_lock wrt ..plt
    mov     rdi,    [rsp]
    mov     r11,    [rdi+16]    ; load ptr for first_available
    mov     r15d,   [r11]       ; load value to callee-saved register
    mov     r14d,   r15d
    add     r14d,   10          ; lasty, incremented y by 10
    mov     r9d,    [rdi]       ; threadsN
    cmp     r9d,    1
    jg .moreThreads
    mov     r14d,   [rsp+60]    ; move height to r14 if 1 thread
    sub     r14d,   [rsp+68]    ; lasty = height-blkSidey
.moreThreads:
    mov     [r11],  r14d        ; update first_available
    mov     rdi,    [rdi+8]
    mov     [rsp+76],   r14d    ; save lasty to stack
    call    pthread_mutex_unlock wrt ..plt

    mov     ecx,    [rsp+60]
    sub     ecx,    [rsp+68]
    sub     ecx,    1           ; height-blkSidey-1
    cmp     r15d,   ecx
    jg .End                     ; No scanlines to process, exit the while-loop

    add     ecx,    1           ; height-blkSidey
    cmp     ecx,    [rsp+76]
    jg .lastDidNotGoOver
    mov     [rsp+76],   ecx
.lastDidNotGoOver:

        mov     r11,    [rsp]
        mov     r12,    [r11+56]
        mov     r13,    [r11+64]
        mov     [rsp+80],   r12     ; *cache_blk_l
        mov     [rsp+88],   r13     ; *cache_blk_r

        mov     r9d,    [rsp+56]
        imul    r9d,    [rsp+52]
        add     r12,    r9
        add     r13,    r9
        mov     [rsp+96],   r12     ; *left rcp_devs
        mov     [rsp+104],  r13     ; right rcp_devs

    .SCANLINE:
        ; Cache data
        mov     r9,     [rsp]
        mov     rdi,    [r9+32]     ; *greyImage0
        mov     esi,    r15d        ; scanLine
        mov     rdx,    [rsp+80]    ; *cache_blk_l
        mov     ecx,    [rsp+56]    ; width
        mov     r8d,    [rsp+40]    ; bx
        mov     r9d,    [rsp+44]    ; by
        call scanline_cacheBlkData_sse
        mov     r9,     [rsp]
        mov     rdi,    [r9+40]     ; *greyImage1
        mov     esi,    r15d        ; scanLine
        mov     rdx,    [rsp+88]    ; *cache_blk_r
        mov     ecx,    [rsp+56]    ; width
        mov     r8d,    [rsp+40]    ; bx
        mov     r9d,    [rsp+44]    ; by
        call scanline_cacheBlkData_sse

        ; Init dmap2 cross-correlation compare-values
        movaps  xmm0,   [rsp+112]   ; load FLT_MAX
        mov     rsi,    [rsp]
        mov     rsi,    [rsi+72]    ; start ptr and also iterator
        mov     [rsp+24],   rsi     ; save start pos to stack
        mov     ecx,    [rsp+56]
        lea     rcx,    [rsi+rcx*4] ; end ptr
        sub     rcx,    12          ; Adjust end for a 4-wide loop
        .INITIALIZE_CCORR_dmap2x4:
            movaps  [rsi],  xmm0
            add     rsi,    16
            cmp     rsi,    rcx
            jl .INITIALIZE_CCORR_dmap2x4
        add     rcx,    12
        cmp     rsi,    rcx
        jge .SKIP_INITx1
        .INITIALIZE_CCORR_dmap2x1:
            movss   [rsi],  xmm0
            add     rsi,    4
            cmp     rsi,    rcx
            jl .INITIALIZE_CCORR_dmap2x1
        .SKIP_INITx1:

        ; Iterate through pixels in a scanline
        ;--------------------------------------

        ; Calculate displacements starting position ptr
        mov     r11,    [rsp]
        mov     rbx,    [r11+80]        ; *displacements
        mov     ecx,    [rsp+56]
        imul    ecx,    r15d            ; scanLine*width
        shl     ecx,    2               ; sizeof(short)*scanLine*width*2
        add     rcx,    rbx             ; &displacements[scanline][0]
        mov     r14d,   [rsp+64]
        .xITER:
            ; rcx + sizeof(short)*x*2
            movzx   eax,    WORD [rcx+r14*4]    ; Zero-extended move, loads d.
            movzx   ebx,    WORD [rcx+r14*4+2]  ; dlim

            movss   xmm15,  [rsp+112]   ; load FLT_MAX

            mov     r12d,   [rsp+52]
            imul    r12d,   r14d        ; x*blkStride
            mov     r13d,   r12d
            add     r12,    [rsp+80]    ; left block x start ptr
            add     r13,    [rsp+88]    ; part of right block start calculation

            mov     r8,     [rsp+96]
            lea     r8,     [r8+r14*4]
            movss   xmm13,  [r8]        ; left rcp_deviation

            ; Iteration count >=1
            .dITER:
                mov     rdi,    r12
                mov     rdx,    r12
                mov     ebp,    [rsp+48]
                add     rdx,    rbp

                mov     rsi,    r13
                mov     r8d,    eax
                imul    r8d,    [rsp+52]
                sub     rsi,    r8          ; right block x start ptr

                mov     r9d,    r14d
                sub     r9d,    eax
                mov     r8,     [rsp+104]
                lea     r8,     [r8+r9*4]
                movss   xmm14,  [r8]        ; right rcp_deviation

                ; accumulators
                xorps   xmm0,   xmm0
                xorps   xmm1,   xmm1
                xorps   xmm2,   xmm2
                xorps   xmm3,   xmm3
                sub     rdx,    60
                cmp     rdi,    rdx
                jge .SKIP_4x4
                .ELEMENTSx4x4:
                    movaps  xmm4,   [rdi]
                    mulps   xmm4,   [rsi]
                    addps   xmm0,   xmm4
                    movaps  xmm5,   [rdi+16]
                    mulps   xmm5,   [rsi+16]
                    addps   xmm1,   xmm5
                    movaps  xmm6,   [rdi+32]
                    mulps   xmm6,   [rsi+32]
                    addps   xmm2,   xmm6
                    movaps  xmm7,   [rdi+48]
                    mulps   xmm7,   [rsi+48]
                    addps   xmm3,   xmm7
                    add     rdi,    64
                    add     rsi,    64
                    cmp     rdi,    rdx
                    jl .ELEMENTSx4x4
                .SKIP_4x4:
                add     rdx,    48
                cmp     rdi,    rdx
                jge .SKIP_4
                .ELEMENTSx4:
                    movaps  xmm4,   [rdi]
                    mulps   xmm4,   [rsi]
                    addps   xmm0,   xmm4
                    add     rdi,    16
                    add     rsi,    16
                    cmp     rdi,    rdx
                    jl .ELEMENTSx4
                .SKIP_4:
                add     rdx,    12
                .ELEMENTSx1:                ; Number of elements is odd,
                    movss   xmm4,   [rdi]   ; so always runs atleast once
                    mulss   xmm4,   [rsi]
                    addss   xmm0,   xmm4
                    add     rdi,    4
                    add     rsi,    4
                    cmp     rdi,    rdx
                    jl .ELEMENTSx1
                movaps  xmm7,   xmm13       ; Multiply rcp_dev's together
                mulss   xmm7,   xmm14

                addps   xmm2,   xmm3
                addps   xmm0,   xmm1
                addps   xmm0,   xmm2
                haddps  xmm0,   xmm0
                haddps  xmm0,   xmm0
                mulss   xmm0,   xmm7

                ucomiss xmm0,   xmm15
                jb .SKIP_SAVE_CURRENT_d_left
                mov     r10d,   eax
                movaps  xmm15,  xmm0
                .SKIP_SAVE_CURRENT_d_left:
                mov     r8,     [rsp+24]
                shl     eax,    2
                sub     r8,     rax
                shr     eax,    2
                movss   xmm7,   [r8+r14*4]
                ucomiss xmm0,   xmm7
                jb .SKIP_SAVE_CURRENT_d_right
                movss   [r8+r14*4], xmm0
                mov     r8d,    r15d
                imul    r8d,    [rsp+56]
                add     r8,     [rsp+16]
                add     r8,     r14
                sub     r8,     rax
                mov     BYTE [r8],  al
                .SKIP_SAVE_CURRENT_d_right:

                add     eax,    1
                cmp     eax,    ebx
                jle .dITER                  ; d less, or equal to dlim
            mov     r8d,    r15d
            imul    r8d,    [rsp+56]
            add     r8,     [rsp+8]
            add     r8,     r14
            mov     BYTE [r8],  r10b

            add     r14d,   1
            cmp     r14d,   [rsp+72]
            jl .xITER

        add     r15d,   1
        cmp     r15d,   [rsp+76]
        jl .SCANLINE

    jmp .whileTop

.End:
    add     rsp,    136
    pop     r12
    pop     r13
    pop     r14
    pop     r15
    pop     rbp
    pop     rbx
    ret

;--------------------------------------------------------------------------------
;void scanline_cacheBlkData(float *img, unsigned int scanline, float *cacheData,
;                           unsigned int width, unsigned int bx, unsigned int by)
;--------------------------------------------------------------------------------
; params (rdi, rsi, rdx, rcx, r8, r9)
; Callee saved: rbx, rbp, r12-r15
scanline_cacheBlkData_sse:

    push    rbx
    push    rbp
    push    r15
    push    r14
    push    r13
    push    r12
    sub     rsp,    120

    ; If width is less than bx, quit.
    cmp     ecx,    r8d
    jl .End

    ; Calculate img-base (Starting position for part of the image)
    mov     eax,    r9d
    shr     eax,    1
    sub     esi,    eax
    imul    esi,    ecx             ; Lazy, just ignoring that params are unsigned
    lea     rdi,    [rdi+rsi*4]
    ; rsi is free for use

    ; Calculate blkStride: (((bx*by)+7)*4/32)*32
    ; Aligns to 32 byte boundary
    mov     eax,    r8d
    imul    eax,    r9d
    add     eax,    7
    shl     eax,    2
    and     eax,    0xffffffe0
    mov     [rsp+12],   eax     ; blkStride (in bytes)

    ; Save float number 1.0f to stack
    mov     eax,        0x3f800000
    mov     [rsp+16],   eax

    ; clear first width*4 bytes of cacheData
    xorps   xmm0,   xmm0
    mov     rax,    rdx
    mov     r10d,   ecx
    shl     r10d,   2
    add     r10,    rdx
    sub     r10,    12      ; minus 3 floats

    cmp     rax,    r10
    jge .SKIPCLEAR16
    .CLEAR16BYTES:
        movaps  [rax],  xmm0    ; zero 16 bytes, *cacheData must be aligned
        add     rax,    16
        cmp     rax,    r10
        jl .CLEAR16BYTES
.SKIPCLEAR16:

    add     r10,    12
    cmp     rax,    r10
    jge .SKIPCLEAR4
    .CLEAR4BYTES:
        movss   [rax],  xmm0    ; zero 4 bytes
        add     rax,    4
        cmp     rax,    r10
        jl .CLEAR4BYTES
.SKIPCLEAR4:

; Add columns together
; Assume variable by is non-zero
    xor     ebp,    ebp     ; y-loop counter
    .yTOP:

        mov     r10d,   ecx
        shl     r10d,   2
        add     r10,    rdx
        sub     r10,    12      ; minus 3 floats

        mov     rax,    rdx         ; start of cacheData
        mov     esi,    ecx
        imul    esi,    ebp         ; width*y
        lea     rbx,    [rdi+rsi*4] ; current image-line ptr

        cmp     rax,    r10
        jge .SKIP_ADD4f
        .ADD4f:
            movups  xmm0,   [rbx]
            addps   xmm0,   [rax]
            movaps  [rax],  xmm0
            add     rax,    16
            add     rbx,    16
            cmp     rax,    r10
            jl .ADD4f
        .SKIP_ADD4f:

        add     r10,    12
        cmp     rax,    r10
        jge .SKIP_ADD1f
        .ADD1f:
            movss   xmm0,   [rbx]
            addss   xmm0,   [rax]
            movss   [rax],  xmm0
            add     rax,    4
            add     rbx,    4
            cmp     rax,    r10
            jl .ADD1f
        .SKIP_ADD1f:

        add     ebp,    1
        cmp     ebp,    r9d
        jl .yTOP

; Mean values for blocks
    mov     eax,    [rsp+12]
    imul    eax,    ecx         ; blkMean-offset in cacheData (blkStride*width)

    add     rax,    rdx
    mov     r10d,   r8d
    shr     r10d,   1
    lea     rax,    [rax+r10*4]     ; first position for written blkMean-value

    mov     ebp,    r8d
    sub     ebp,    1
    shl     ebp,    2           ; amount to subtract from cacheData-ptr when
                                ; going for next iteration

    mov     r11d,   ecx
    shl     r10d,   1           ; bxSide*2
    sub     r11d,   r10d
    shl     r11d,   2
    add     r11,    rax         ; .nextBlkMean ending pointer

    movss       xmm2,   [rsp+16]; float_one
    cvtsi2ss    xmm3,   r8d
    cvtsi2ss    xmm4,   r9d
    mulss       xmm3,   xmm4
    divss       xmm2,   xmm3    ; 1.0/(bx*by)

    mov     rbx,    rdx         ; *cacheData

    lea     r10,    [rbx+r8*4]
    xorps   xmm0,   xmm0    ; accumulator

    .ADDBX_ROW:
        addss   xmm0,   [rbx]
        add     rbx,    4
        cmp     rbx,    r10
        jl .ADDBX_ROW

    movaps  xmm1,   xmm0    ; copy for next stage
    mulss   xmm0,   xmm2
    sub     rbx,    rbp     ; subtract bx-1 for next iteration
    movss   [rax],  xmm0

    add     rax,    4       ; assume there is block to calculate
    .SUB1ADD1:              ; in other words width >= bx+1
        subss   xmm1,   [rbx-4]
        addss   xmm1,   [rbx+rbp]
        movaps  xmm0,   xmm1
        mulss   xmm0,   xmm2
        movss   [rax],  xmm0
        add     rbx,    4
        add     rax,    4
        cmp     rax,    r11
        jl .SUB1ADD1


; Subtract mean from block element and save them,
; calculate reciprocal of deviations
    mov     rax,    rdx
    mov     ebx,    r8d
    shr     ebx,    1
    add     rax,    rbx     ; cacheData+bxSide
    mov     [rsp],  ebx     ; bxSide is iterator start for iTop-loop
    mov     [rsp+8],    ebx ; Avoid calculating bxSide on loops

    mov     esi,    ecx
    sub     esi,    ebx
    mov     [rsp+4],    esi ; width-bxSide, iTop-loop end

    mov     r12d,   ecx
    sub     r12d,   r8d     ; width-bx

    ; Starting from i=bxSide
    .iTOP:
        mov     r11d,   ecx
        imul    r11d,   [rsp+12]
        mov     ebx,    [rsp]
        lea     r11,    [r11+rbx*4] ; offset for ptr to mean

        movss   xmm7,   [rdx+r11]   ; mean
        shufps  xmm7,   xmm7,   0x0 ; broadcast lowest vector element to others

        ; accumulator
        xorps   xmm6,   xmm6

        mov     esi,    [rsp]
        imul    esi,    [rsp+12]
        add     rsi,    rdx         ; cacheData[i*blkStride]

        mov     ebx,    [rsp]
        sub     ebx,    [rsp+8]
        lea     r10,    [rdi+rbx*4] ; start of innermost loop
        lea     r15,    [r10+r8*4]  ; stop of innermost loop

        xor     r11d,   r11d        ; y counter
        ; 1-block wide
        .y_0_BY:
            sub     r15,    12
            cmp     r10,    r15
            jge .SKIP4              ; Skips, if bx=3
            .subMeanAndAddDevs4:
                movups  xmm0,   [r10]
                subps   xmm0,   xmm7
                movups  [rsi],  xmm0
                mulps   xmm0,   xmm0
                addps   xmm6,   xmm0
                add     r10,    16
                add     rsi,    16
                cmp     r10,    r15
                jl .subMeanAndAddDevs4

            .SKIP4:

            add     r15,    12          ; Adjust loop control for 1-wide

            .subMeanAndAddDevs1:        ; always executes atleast once
                movss   xmm0,   [r10]
                subss   xmm0,   xmm7
                movss   [rsi],  xmm0
                mulss   xmm0,   xmm0
                addss   xmm6,  xmm0    ; can use same accumulator
                add     r10,    4
                add     rsi,    4
                cmp     r10,    r15
                jl .subMeanAndAddDevs1

            ; update innermost loop iteration limits
            lea     r10,    [r10+r12*4] ; add (width-bx)*4
            lea     r15,    [r10+r8*4]  ; r10 + bx*4

            add     r11d,   1
            cmp     r11d,   r9d         ; r9 = by
            jl .y_0_BY

        ; Serialize accumulator
        ;addps   xmm6,   xmm11
        movaps  xmm0,   xmm6
        shufps  xmm0,   xmm0,   0xe
        addps   xmm0,   xmm6
        movaps  xmm1,   xmm0
        shufps  xmm1,   xmm1,   0x1
        addss   xmm0,   xmm1

        sqrtss  xmm0,   xmm0
        mov     ebx,    [rsp]
        movss   xmm1,   [rsp+16]    ; float_one
        divss   xmm1,   xmm0
        mov     eax,    [rsp+12]
        imul    eax,    ecx
        mov     r14d,   ebx
        shl     r14d,   2
        add     eax,    r14d
        add     rax,    rdx
        movss   [rax],  xmm1

        add     ebx,    1
        mov     [rsp],  ebx
        cmp     ebx,    [rsp+4]
        jl .iTOP

.End:
    add     rsp,    120
    pop     r12
    pop     r13
    pop     r14
    pop     r15
    pop     rbp
    pop     rbx
    ret

;-----------------------------------
;void *blendWorker(void *threadData)
;-----------------------------------
; params (rdi)
; Callee saved: rbx, rbp, r12-r15
blendWorker_sse2:

    push    rbx
    push    r15
    push    r14

    sub     rsp,    32

    mov     [rsp],  rdi
    mov     ecx,    [rdi+32]    ; read width
    mov     [rsp+8],    ecx     ; save width to stack
    mov     edx,    [rdi+36]
    mov     [rsp+12],   edx     ; height

    ;    rgbConv weights
    mov     eax,        0x00000000
    mov     [rsp+28],   eax
    mov     eax,        0x3d93dd98
    mov     [rsp+24],   eax
    mov     eax,        0x3f371759
    mov     [rsp+20],   eax
    mov     eax,        0x3e59b3d0
    mov     [rsp+16],   eax

    ; NOTE c-version divides heights by 4, asm does not, do not mix c- and asm-threads
    .whileTop:
    mov     r11,    [rsp]
    mov     rdi,    [r11+8]     ; load ptr for mutex
    call    pthread_mutex_lock wrt ..plt
    mov     rdi,    [rsp]
    mov     r11,    [rdi+16]    ; load ptr for first_available
    mov     r15d,   [r11]       ; load value to callee-saved register
    mov     r14d,   r15d        ; y saved to callee-saved register
    add     r15d,   128         ; lasty
    mov     r9d,    [rdi]       ; threadsN
    cmp     r9d,    1
    je .lasty_equalto_height
.modifiedR15:
    mov     [r11],  r15d        ; update first_available
    mov     rdi,    [rdi+8]
    call    pthread_mutex_unlock wrt ..plt

    mov     ecx,    [rsp+12]
    sub     ecx,    5
    cmp     r14d,   ecx
    jg .End                     ; No scanlines to process, exit the while-loop

    add     ecx,    2           ; height-3
    cmp     r15d,   ecx
    jg .EqualHeightminus3
.modifiedR15again:

    cmp     r14d,   r15d        ; FIXME: Almost identical check as few lines above.
    jg .End                     ; No work on entry, something gone wrong?
    mov     rdi,    [rsp]
    mov     r8,     [rdi+24]    ; src-pointer

    pxor        xmm7,   xmm7    ; zero vector
    mov     ecx,    [rsp+8]
    shl     ecx,    2           ; line-stride
    mov     edx,    ecx
    imul    edx,    3           ; line-stride*3

    .yTop:
        mov     ebx,    [rsp+8] ; width

        mov     r11d,   r14d
        shl     r11d,   2
        imul    ebx,    r11d
        add     rbx,    r8          ; src position ptr

        mov     r10d,    ecx
        add     r10,    rbx
        sub     r10,    12          ; check x4 iterations against this

        mov     rax,    [rdi+40]    ; dst-pointer
        mov     r9d,    [rsp+8]
        and     r9d,    0xfffffffc  ; equivalent to (width/4)*4, sets 2 lowest bits to zero
        mov     r11d,   r14d
        shr     r11d,   0x2         ; current dst-y
        imul    r9d,    r11d
        add     rax,    r9          ; dst-pointer for a line

        cmp     rbx,    r10
        jge .skip_x4

        ALIGN 16
        ; Reads 4x4 32bit pixels wide part of the image
        .x4Top:
            movups  xmm0,   [rbx]
            movups  xmm1,   [rbx+rcx]       ; +stride
            movups  xmm2,   [rbx+rcx*2]     ; +stride*2
            movups  xmm3,   [rbx+rdx]       ; +stride*3

            ; Convert 8-bit values to 16-bit
            movaps      xmm4,   xmm0
            punpcklbw   xmm0,   xmm7
            punpckhbw   xmm4,   xmm7
            movaps      xmm5,   xmm1
            punpcklbw   xmm1,   xmm7
            punpckhbw   xmm5,   xmm7

            paddw       xmm0,   xmm1
            paddw       xmm4,   xmm5
            paddw       xmm0,   xmm4

            movaps      xmm1,   xmm2
            punpcklbw   xmm1,   xmm7
            punpckhbw   xmm2,   xmm7
            movaps      xmm5,   xmm3
            punpcklbw   xmm3,   xmm7
            punpckhbw   xmm5,   xmm7

            paddw       xmm1,   xmm3
            paddw       xmm2,   xmm5
            paddw       xmm1,   xmm2

            paddw       xmm0,   xmm1
            movaps      xmm1,   xmm0
            pshufd      xmm1,   xmm1,   0xe     ; move upper register to lower portion
            paddw       xmm0,   xmm1            ; only lower half has meaninful values
            psrlw       xmm0,   0x4             ; Divide by 16

            punpcklwd   xmm0,   xmm7
            cvtdq2ps    xmm0,   xmm0            ; Convert to floating point
            mulps       xmm0,   [rsp+16]        ; rgbConv weights
            movaps      xmm1,   xmm0
            movaps      xmm2,   xmm0
            shufps      xmm1,   xmm1,   0x1
            shufps      xmm2,   xmm2,   0x2
            addss       xmm0,   xmm1
            addss       xmm0,   xmm2

            movss   [rax],  xmm0

            add     rbx,    16
            add     rax,    4

            cmp     rbx,    r10
            jl .x4Top
        .skip_x4:

        add     r14d,   4
        cmp     r14d,   r15d
        jl .yTop

        jmp .whileTop


.End:
    add     rsp,    32

    pop     r14
    pop     r15
    pop     rbx
    ret

.lasty_equalto_height:
    mov     r15d,   [rdi+36]    ; move height to r15
    jmp .modifiedR15

.EqualHeightminus3:
    mov     r15d,   ecx
    jmp .modifiedR15again

;------------------------------------------------------------
;float *blend_2x2(float *img, unsigned int w, unsigned int h)
;------------------------------------------------------------
; params(rdi, rsi, rdx)
; Callee saved: rbx, rbp, r12-r15
; return rax
blend_2x2_sse3:

    push    rbx
    push    rbp
    sub     rsp,    56
    mov     rax,    0
    ; Test image for dimensions and return NULL if either is zero or one.
    cmp    rsi,    2
    jl .End
    cmp    rdx,    2
    jl .End

    mov     [rsp+24],   rdi     ; src ptr
    mov     [rsp+16],   rsi     ; width
    mov     [rsp+8],   rdx      ; height

    ; Create 4 elem vector of value 1.0f/4.0f in divisible by 16 stack-address
    mov     eax,        0x3e800000
    mov     [rsp+32],   eax
    mov     [rsp+36],   eax
    mov     [rsp+40],   eax
    mov     [rsp+44],   eax

    ; call posix_memalign
    imul    rdx,        rsi
    shl     rdx,        2
    mov     rsi,        16
    mov     rdi,        rsp     ; posix_memalign saves pointer to [rsp]
    call    posix_memalign wrt ..plt

    ; entry for y-loop
    mov     rdi,    [rsp+24]
    xor     r9d,    r9d
    .y_top:

        mov     rcx,    [rsp+16]
        mov     rdx,    [rsp+16]
        imul    rcx,    r9
        shl     rcx,    2
        add     rcx,    rdi         ; 1st line current position pointer
        lea     rdx,    [rcx+rdx*4] ; 2nd line current position pointer

        mov     rbx,    rdx
        sub     rbx,    28          ; 2nd line minus 7 floats to avoid overread in x8-loop

        mov     rax,    [rsp+16]
        shr     rax,    1
        mov     rbp,    r9
        shr     rbp,    1
        imul    rax,    rbp
        shl     rax,    2
        add     rax,    [rsp]       ; rax is dst current position pointer

        ; x8-loop entry
        mov     r8,     [rsp+16]
        cmp     r8,     8
        jl .skip_x8

        ALIGN 16
        .x8_top:
            movups  xmm0,   [rcx]
            movups  xmm1,   [rcx+16]
            movups  xmm2,   [rdx]
            movups  xmm3,   [rdx+16]

            addps   xmm0,   xmm2
            addps   xmm1,   xmm3
            haddps  xmm0,   xmm1
            mulps   xmm0,   [rsp+32]  ; Load reciprocal (1.0f/4.0f)

            movups  [rax],  xmm0

            add     rcx,    32
            add     rdx,    32
            add     rax,    16
            cmp     rcx,    rbx
            jl  .x8_top

            ; x2-loop entry from x8-loop
            sub     rcx,    24
            sub     rdx,    24
            sub     rax,    12
            cmp     rcx,    rbx
            jge .skip_x2

        .skip_x8:
            add     rbx,    24      ; re-adjust loop-ending
        ALIGN 16
        ; in case of skipping x8_loop it was earlier guaranteed that atleast
        ; 2 pixels remain to be processed
        .x2_top:
            movss   xmm0,   [rcx]
            movss   xmm1,   [rcx+4]
            movss   xmm2,   [rdx]
            movss   xmm3,   [rdx+4]

            addss   xmm0,   xmm1
            addss   xmm2,   xmm3
            addss   xmm0,   xmm2
            mulss   xmm0,   [rsp+32]    ; Load reciprocal (1.0f/4.0f)

            movss  [rax],  xmm0

            add     rcx,    8
            add     rdx,    8
            add     rax,    4
            cmp     rcx,    rbx
            jl .x2_top

    .skip_x2:
        add     r9d,    2
        cmp     r9d,    [rsp+8]
        jl .y_top

    mov     rax,    [rsp]
.End:
    add     rsp,    56
    pop     rbp
    pop     rbx
    ret
