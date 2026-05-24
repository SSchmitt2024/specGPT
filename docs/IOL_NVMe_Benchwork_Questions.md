# UNH-IOL NVMe Benchwork Study Questions

Source: **UNH-IOL NVMe Testing Service – Test Plan for NVM Command Set Conformance, v24.0 (Aug 1, 2025)**
Target spec: NVMe NVM Command Set Specification 1.1 + NVMe Base Specification 2.2

Questions are pulled from real Test Procedures (TPs) in the conformance test plan and tagged by difficulty:
**[E]asy** – definitions, fields, single-step lookups
**[M]edium** – multi-step procedures, status codes, conditional flows
**[H]ard** – cross-feature interactions, edge cases, version-dependent behavior

---

## Group 1 – Admin Command Set

### Test 1.1 – Identify Command

**Q1. [E]** What CNS value is used to retrieve the Identify Controller Data Structure, and what is the size of the returned structure?
**A.** CNS = `01h`. The Identify data structure is **4096 bytes**. It is posted to the memory buffer indicated in PRP Entry 1/2 and Command Dword 10, and a completion entry is posted to the Admin Completion Queue.

**Q2. [E]** Which CNS value returns the Namespace List of active NSIDs?
**A.** CNS = `02h`, with `CDW1.NSID = 0` to start the list from the lowest NSID.

**Q3. [M]** When issuing an Identify with `CNS=00h` against an **inactive** namespace, what must the returned data structure look like?
**A.** The returned Identify Namespace data structure must be **zero-filled**, and all reserved fields must be 0.

**Q4. [M]** The Identify Namespace data structure reports `THINP=0` in NSFEAT. What constraint must hold between Namespace Capacity (NCAP) and Namespace Size (NSZE)?
**A.** When `THINP=0` (thin provisioning not supported), `NCAP` must equal `NSZE`. When `THINP=1`, the controller must track allocated blocks via the NUSE field.

**Q5. [M]** In an Identify Namespace response, what does the `NLBAF` field tell you about which LBA Format descriptors are valid?
**A.** `NLBAF` (Number of LBA Formats) is 0's based — formats 0 through NLBAF are valid. Any LBA Format Support descriptors at indices beyond NLBAF (i.e. LBAFn+i) must be zero.

**Q6. [M]** ASCII string fields like Serial Number (SN), Model Number (MN), and Firmware Revision (FR) — how must they be formatted in the Identify Controller data structure?
**A.** Left-justified and **padded to the right with ASCII space (`20h`)** — not null-terminated, not zero-padded.

**Q7. [H]** A DUT claims support for NVMe 1.4. Identify Controller returns `CNTRLTYPE = 0`. Pass or fail, and why?
**A.** **Fail.** For NVMe 1.4 or later, CNTRLTYPE must not be 0 — it must identify the controller type (I/O controller, discovery, admin, etc.).

**Q8. [H]** Identify Controller reports `FNA` bit 3 set to 1. What constraints must hold for FNA bits 0 and 1?
**A.** When FNA bit 3 = 1, FNA bits 0 and 1 must both be cleared to 0.

**Q9. [H]** When does the controller need to support CNS values `10h–13h`?
**A.** Whenever Namespace Management is supported — i.e., OACS bit 3 = 1. CNS `10h` (Namespace ID List), `11h` (Allocated Namespace IDS), `12h` (Namespace Attached Controller List), `13h` (Controller List) all become mandatory.

**Q10. [H]** Identify response for CNS `03h` (Namespace Identification Descriptor) — what conditions trigger a Type 3 (UUID) entry?
**A.** When **both** NGUID and EUI64 are set to 0 in the Identify Namespace data structure, the Namespace Identification Descriptor must report a type-3 (NUUID) descriptor. The controller must not return multiple descriptors with the same NIDT.

---

### Test 1.2 – Set/Get Features

**Q11. [E]** What does `SEL = 000b` mean in a Get Features command?
**A.** Return the **Current** value of the feature.

**Q12. [E]** What are the four SEL encodings for Get Features?
**A.** `000b` = Current, `001b` = Default, `010b` = Saved, `011b` = Supported Capabilities.

**Q13. [M]** A host sends `Set Features` for FID `07h` (Number of Queues) **after** I/O queues have been created. What completion status is expected?
**A.** **Command Sequence Error.** Number-of-Queues can only be set before any I/O queues exist on the controller (Test 1.4 Case 11).

**Q14. [H]** You issue Get Features for Sanitize Config (FID `17h`) with `SEL=011b` and bit 0 of Dword 0 is cleared to 0. What does this tell you about the default NODRM value?
**A.** Sanitize Config is **not savable** — therefore the default value of the NODRM attribute must be cleared to 0 (Sanitize Test 1.17 Case 3).

---

### Test 1.3 – Get Log Page

**Q15. [E]** Name the mandatory log pages a controller must support (LIDs 00h–03h, plus the two Effects logs).
**A.** `00h` Supported Log Pages, `01h` Error Information, `02h` SMART/Health, `03h` Firmware Slot Information, `12h` Feature Identifiers Supported and Effects, `13h` NVMe-MI Commands Supported and Effects (only if NVMe-MI Send/Recv supported).

**Q16. [E]** What's the LID for the Sanitize Status log page?
**A.** `81h`.

**Q17. [M]** A host issues Get Log Page with a vendor-specific LID (range `C0h–FFh`) that the controller doesn't support. What status code is expected?
**A.** **Invalid Log Page (`09h`).**

**Q18. [M]** What range of LIDs is reserved for NVMe over Fabrics?
**A.** `71h–7Fh`.

**Q19. [H]** If MDTS is non-zero and a host sets `NUMDU/NUMDL` (or NUMD on older drives) larger than MDTS in a Get Log Page command, what happens?
**A.** The controller returns **Invalid Field in Command** — the requested data transfer exceeds MDTS.

**Q20. [H]** For a DUT supporting NVMe 2.0 or higher, may LID `00h` (Supported Log Pages) be returned with status "Invalid Log Page"?
**A.** **No.** LID 00h is mandatory in 2.0+; if the DUT does not support it, the test fails. For 1.3 and earlier, LID 00h was reserved and "Invalid Field in Command" was also acceptable.

---

### Test 1.4 – Create/Delete I/O Submission and Completion Queues

**Q21. [E]** What must the host do before deleting an I/O Completion Queue?
**A.** **Delete all associated I/O Submission Queues first.** Deleting the CQ before its SQ produces status **Invalid Queue Deletion**.

**Q22. [E]** What status is returned when a Create I/O Completion Queue is sent with `QID=0`?
**A.** **Invalid Queue Identifier (`01h`)** — QID 0 is reserved for the Admin queue pair.

**Q23. [M]** Create I/O Submission Queue with `QSIZE=0`. What happens?
**A.** Status = **Invalid Queue Size**. Same result if QSIZE > CAP.MQES.

**Q24. [M]** If `CAP.CQR = 1` and a Create I/O Submission Queue command sets `PC=0`, what error is returned?
**A.** **Invalid Field in Command** — when contiguous queues are required, PC must be 1.

**Q25. [M]** Create I/O Submission Queue references a CQID that is within the controller-supported range but the matching CQ has not been created. Status?
**A.** **Completion Queue Invalid (`00h`)** (distinct from Invalid Queue Identifier).

**Q26. [H]** What's the difference between status `00h` and `01h` failure paths when creating an I/O SQ with a bad CQID?
**A.** `00h` Completion Queue Invalid → CQID is in valid range but no CQ exists there. `01h` Invalid Queue Identifier → CQID is outside the supported NCQA range (or is 0).

**Q27. [H]** The Create I/O Completion Queue command is sent with an Interrupt Vector greater than `MSICAP.MC.MME` or `MSIXCAP.MXC.TS`. Expected status?
**A.** **Invalid Interrupt Vector.**

---

### Test 1.5 – Abort Command

**Q28. [E]** Where is the "abort success" indication located in the completion queue entry of the Abort command?
**A.** **Bit 0 of Dword 0** — cleared to 0 means the command was aborted; set to 1 means the abort did not take effect.

**Q29. [M]** When the abort succeeds, what status code must the aborted command itself return, and in what order are the two CQEs posted?
**A.** The aborted command reports status **Command Abort Requested (`07h`)**. Its CQE must be posted to the I/O Completion Queue **before** the Abort command's CQE is posted to the Admin Completion Queue.

**Q30. [H]** What limits the number of outstanding Abort commands a host may have?
**A.** The **Abort Command Limit (ACL)** field of the Identify Controller Data Structure.

---

### Test 1.6 – Format NVM

**Q31. [M]** A DST is in progress on a specific NSID. The host sends Format NVM with the same NSID. What must happen to the DST?
**A.** The Device Self-Test operation is **aborted by the Format NVM**. The DST log will reflect the abort.

**Q32. [H]** DST is running with `NSID=FFFFFFFFh`. The host issues Format NVM with `NSID=FFFFFFFFh`. Expected behavior?
**A.** The broadcast Format aborts the broadcast DST (Test 1.10 Case 8 / 1.11 Case 7).

---

### Test 1.7 – Asynchronous Events

**Q33. [E]** What is the purpose of the Asynchronous Event Request command?
**A.** The host pre-posts AERs so the controller has a completion slot available when an event occurs. The controller does not return a CQE until an asynchronous event triggers.

**Q34. [M]** What happens when more AERs are submitted than the controller's Asynchronous Event Request Limit (AERL) allows?
**A.** The excess AER is aborted with **Asynchronous Event Request Limit Exceeded** (Test 1.7 Case 2 / 5.4 Case 2).

**Q35. [H]** What async event must be generated when the controller enters the Sanitize Media Verification State?
**A.** A Sanitize "Operation Entered Media Verification State" notice (Test 1.7 Case 6) — additionally surfaced in the Persistent Event Log as event type `0Eh`.

---

### Test 1.10 / 1.11 – Device Self-Test

**Q36. [E]** Which OACS bit indicates support for the Device Self-Test command?
**A.** **OACS bit 4.**

**Q37. [E]** What is the LID for the Device Self-Test log page?
**A.** `06h`.

**Q38. [M]** A second DST command is sent while one is already in progress. What status is returned for the second command?
**A.** **Device Self-Test in Progress.**

**Q39. [M]** What STC value selects a short DST? What value selects an extended DST?
**A.** STC = `1h` → Short DST. STC = `2h` → Extended DST.

**Q40. [H]** DST is sent with NSID = an Inactive Namespace. What status do we expect?
**A.** **Invalid Field in Command.** (For a truly invalid NSID: "Invalid Namespace or Format in Command.")

---

### Test 1.17 – Sanitize Command

**Q41. [E]** Which Identify Controller field tells you which Sanitize operations the DUT supports?
**A.** **SANICAP.** Bits 2:0 indicate Crypto Erase / Block Erase / Overwrite support.

**Q42. [M]** A Sanitize is in progress. The host issues a Read. What status does the Read complete with?
**A.** **Sanitize In Progress.** Same applies to Compare, DSM, Write, Write Uncorrectable, Write Zeroes, Verify, and the four Reservation commands while sanitize is active.

**Q43. [M]** A Sanitize is in progress. The host polls the Sanitize Status log page (LID `81h`). What value of `SSTAT` is expected, and what range applies to `SPROG`?
**A.** `SSTAT[2:0] = 010b` (Sanitize in Progress). `SPROG ≠ FFFFh` while the operation is running.

**Q44. [M]** After a Sanitize completes successfully, the host re-reads the Sanitize Status log. What does the log show?
**A.** `SSTAT[2:0] = 001b` (Most recent sanitize completed successfully) and `SPROG = FFFFh`.

**Q45. [H]** SANICAP NDI=1. Host sets `Sanitize Config (FID 17h)` with `NODRM = 0`, then issues Sanitize with `No Deallocate After Sanitize (NDAS) = 1`. What status does the Sanitize complete with?
**A.** **Invalid Field in Command** — when NODRM is 0, the controller is prohibited from honoring NDAS=1.

**Q46. [H]** Host sends a Sanitize against a namespace that is currently Write-Protected. Result?
**A.** Sanitize completes with **Namespace is Write Protected**.

**Q47. [H]** What is `SSI=5h` in the Sanitize Status log page, and how is it reached?
**A.** Sanitize is in the **Media Verification State**. Reached by sending Sanitize with `EMVS=1` and `SANACT=010b` (Block Erase) or `100b` (Crypto Erase). An NVM Subsystem Reset clears this state and sets `MVCNCLD=1`.

---

### Test 1.21 – Command and Feature Lockdown

**Q48. [M]** Which Get Log Page LID surfaces the active Command and Feature Lockdown state?
**A.** `14h`.

---

### Test 1.24 – Keep Alive Timer

**Q49. [E]** What feature ID configures the Keep Alive Timer?
**A.** `0Fh`.

**Q50. [M]** What happens when the Keep Alive Timeout elapses without a Keep Alive command?
**A.** The controller logs a fatal status and (for fabrics) tears down the connection; for PCIe it can trigger CSTS.CFS = 1 depending on the implementation.

---

## Group 2 – NVM Command Set

### Test 2.1 – Compare

**Q51. [E]** Which ONCS bit indicates Compare support?
**A.** ONCS **bit 0**.

**Q52. [M]** Compare a known-written LBA against mismatched data. Status?
**A.** **Compare Failure (`85h`)** in the Media and Data Integrity Errors group.

---

### Test 2.2 – Dataset Management (DSM)

**Q53. [E]** Which ONCS bit indicates DSM (Deallocate) support?
**A.** ONCS **bit 2**.

**Q54. [E]** What is the maximum number of ranges in a single DSM command, and what is the total payload size?
**A.** **256 ranges**, **4096 bytes** total (16 bytes per range × 256).

**Q55. [M]** After deallocating an LBA range with DSM (AD=1), a subsequent Read to that range returns what data?
**A.** Implementation-defined but **deterministic**: all zeros, all ones, or the last data written. The same read repeated must return the same pattern until a new Write occurs to that LBA.

**Q56. [M]** DSM with `NSID=FFFFFFFFh` against a DUT supporting NVMe 1.4+. Expected result?
**A.** **Does not complete successfully** — broadcast NSID isn't allowed for DSM on 1.4+.

**Q57. [H]** What status is returned when `ONCS bit 2 = 0` and a DSM command exceeds DMRL / DMRSL / DMSL?
**A.** **Command Limit Exceeded.** (When `ONCS bit 2 = 1`, the controller must enforce limits but is not required to use this exact status.)

---

### Test 2.3 / 2.4 – Read / Write

**Q58. [E]** What is the data structure used in the Completion Queue Entry (CQE)?
**A.** **16 bytes**: DW0 (command-specific), DW1 (command-specific), DW2 (SQ Head Pointer + SQ Identifier), DW3 (CID + P bit + Status Field).

**Q59. [M]** What status is returned for a Read or Write whose LBA range extends beyond the namespace size?
**A.** **LBA Out of Range (`80h`)** in the Generic Command Status group.

---

### Test 2.5 – Write Uncorrectable

**Q60. [M]** After a Write Uncorrectable to a range of LBAs, a subsequent Read of that range returns what status?
**A.** **Unrecovered Read Error (`81h`)** — Media and Data Integrity Errors group.

---

### Test 2.7 – Write Zeroes

**Q61. [M]** Write Zeroes is issued with `Deallocate (DEAC) = 1`. What is the visible effect compared to DSM-Deallocate?
**A.** The targeted range reads as deterministic zeros and is also deallocated (subject to controller behavior reported in NSFEAT/DLFEAT).

---

### Test 2.8 – Atomicity Parameters

**Q62. [E]** What relationship must always hold between AWUPF and AWUN?
**A.** **AWUPF ≤ AWUN.**

**Q63. [M]** List the required inequalities among AWUN, AWUPF, NAWUN, NAWUPF, NABSN, NABO, NABSPF, NACWU, ACWU when the NVM command set is supported.
**A.**
- AWUPF ≤ AWUN
- NAWUN ≥ AWUN
- NAWUPF ≥ AWUPF
- NAWUPF ≤ NAWUN
- NACWU ≥ ACWU
- NABSN ≥ NAWUN
- NABO ≤ NABSN
- NABO ≤ NABSPF
- NABSPF ≥ NAWUPF

**Q64. [H]** Bit 37 of CAP.CSS = 0 (NVM Command Set **not** supported). What values must AWUN, AWUPF, ACWU report?
**A.** All three must be **set to 0** (Test 2.8 Case 2).

---

### Test 2.9 – AWUN/NAWUN Behavior

**Q65. [H]** With `NABSN = 0` (atomic boundaries unsupported), two overlapping writes A (LBAs 0–3, pattern FFFFh) and B (LBAs 1–4, pattern AAAAh) are submitted. Which final states are conformant?
**A.** Either: (a) LBAs 0–3 = FFFFh and LBA 4 = AAAAh, OR (b) LBA 0 = FFFFh and LBAs 1–4 = AAAAh. Any mix of partial pattern interleaving across a single write is a failure — write atomicity must hold for each write up to AWUN.

---

### Test 2.11 – Verify

**Q66. [M]** What does the Verify command do, and what does it return on success?
**A.** Verify reads the requested LBA range and checks integrity (and PI if enabled) without returning data. On success the completion is `Success`. On integrity failure the appropriate Media and Data Integrity status (e.g. `Unrecovered Read Error`) is returned.

---

### Test 2.12 – Fused Operations

**Q67. [H]** A Compare-and-Write fused operation has the first command (Compare) succeed but the data does not match. What happens to the second command (Write)?
**A.** The Write is **aborted with "Command Aborted Due to Failed Fused Command"**, because the fused pair only proceeds when the Compare succeeds.

---

### Test 2.14 – Copy

**Q68. [M]** Which ONCS bit indicates Copy support?
**A.** ONCS **bit 8**.

**Q69. [H]** A Copy command's source range crosses an atomic boundary in a target namespace that supports descriptor format `00h` only. Is atomicity guaranteed for the destination range?
**A.** **Only up to AWUN/NAWUN per chunk** — Copy does not change atomicity guarantees. Crossing a boundary breaks atomicity.

---

## Group 3 – NVM Features

### Test 3.2 – End-to-End Data Protection

**Q70. [E]** What is "PIL"? Where is it found?
**A.** **Protection Information Location** — the field in the Identify Namespace data structure (DPC field) indicating whether PI is at the start or end of the metadata.

**Q71. [M]** Which Identify Namespace fields advertise the PI types supported?
**A.** **DPC** (Data Protection Capabilities) bits 0–2 indicate support for Type 1 / 2 / 3. DPS field selects which is currently enabled.

**Q72. [H]** A namespace is formatted with **PI Type 1**, metadata 8 bytes, transferred as a contiguous part of the LBA (`MS` enabled). The host issues a Write with `PRACT=1`. What does the controller do?
**A.** The controller **inserts protection information** (Guard CRC + Application Tag + Reference Tag) into the metadata as part of the write. On Read with `PRACT=1`, the controller strips PI after verifying it.

---

### Test 3.3 – Power Management

**Q73. [E]** Which feature identifier selects the active power state?
**A.** FID **`02h`** Power Management. The value is in CDW11 bits 4:0 (Power State).

**Q74. [M]** What is the maximum number of power states reportable, and which Identify Controller field bounds the valid range?
**A.** 32 (PS0–PS31). **NPSS** in the Identify Controller data structure gives the highest supported PS index (0's based, so NPSS = N means PS0..PSN are supported). Power state descriptors beyond NPSS must be zero.

---

### Test 3.4 – Host Memory Buffer

**Q75. [M]** Which feature identifier configures the Host Memory Buffer?
**A.** FID **`0Dh`**.

**Q76. [H]** Host issues Set Features for HMB to **enable** HMB when it is already enabled. Expected behavior?
**A.** The command must complete with status indicating the operation could not proceed — the test verifies the controller correctly rejects re-enabling an already-enabled HMB (Test 3.4 Case 4).

---

### Test 3.13 – Read Recovery Level

**Q77. [E]** Which feature identifier configures Read Recovery Level?
**A.** FID **`12h`**.

---

### Test 3.14 – Asymmetric Namespace Access (ANA)

**Q78. [M]** What log page exposes ANA state? Which LID?
**A.** Asymmetric Namespace Access log page, **LID `0Ch`**.

---

## Group 4 – Controller Registers

### Test 4.1 / 4.2 – CAP.MPSMAX / MPSMIN

**Q79. [E]** What is the formula for the host memory page size from MPSMAX or MPSMIN?
**A.** `Page size = 2^(12 + MPSx)` bytes. Min reportable is 4 KiB (MPSMIN=0), max is 128 MiB (MPSMAX=16).

**Q80. [M]** What relationship must always hold between MPSMAX and MPSMIN?
**A.** MPSMAX ≥ MPSMIN.

---

### Test 4.3 – CAP.CSS

**Q81. [E]** Which CAP.CSS bit indicates NVM Command Set support?
**A.** **Bit 37.**

**Q82. [M]** For a controller advertising NVMe 2.0+, what must be true of CAP.CSS bit 7?
**A.** **Cleared to 0** (it was redefined in 2.0; bit 7 is no longer a valid command-set indicator).

**Q83. [H]** CAP.CSS bit 43 (Controller Supports One or More I/O Command Sets) = 1 and bit 37 (NVM Command Set) = 1. Is this contradictory?
**A.** **No.** Bit 37 must still be set even when bit 43 is set, as long as the NVM Command Set is one of the supported sets.

---

### Test 4.4 – CAP.DSTRD

**Q84. [E]** What is the formula for the doorbell stride from CAP.DSTRD, and what value packs doorbells contiguously?
**A.** `Stride = 2^(2 + DSTRD)` bytes. `DSTRD = 0` → 4-byte stride (contiguous).

---

### Test 4.5 – CAP.TO

**Q85. [E]** What units does CAP.TO use, and what does it bound?
**A.** **500 ms units.** It is the worst-case time host software must wait for CSTS.RDY to transition after toggling CC.EN.

**Q86. [M]** The host sets CC.EN=1 but CSTS.RDY never goes to 1 within `CAP.TO × 500 ms`. What's the verdict?
**A.** **Test fails** — the controller violated its advertised ready timeout.

---

### Test 4.8 – CAP.MQES

**Q87. [E]** What's the minimum legal CAP.MQES value and what does it indicate?
**A.** **MQES = 1h**, meaning **2 entries minimum** (0's based). Hosts must not create queues larger than MQES+1 entries.

---

### Test 4.12 – CC.SHN (Shutdown Notification)

**Q88. [M]** What are the three CC.SHN encodings, and how does the host confirm shutdown completed?
**A.** `00b` = No notification, `01b` = Normal shutdown, `10b` = Abrupt shutdown. The host polls **CSTS.SHST** until it reads `10b` (Shutdown processing complete).

**Q89. [H]** During normal shutdown the controller must complete which kinds of operations before signaling complete?
**A.** Flush all outstanding writes/cached data to non-volatile media, complete any in-flight commands per spec, then set CSTS.SHST = `10b`.

---

### Test 4.15 – CC.EN

**Q90. [M]** The host writes CC.EN = 0 → 1. What must CSTS.RDY do, and within what time?
**A.** CSTS.RDY must transition 0 → 1 within `CAP.TO × 500 ms`.

---

### Test 4.17 – CSTS.CFS

**Q91. [E]** What does CSTS.CFS indicate?
**A.** **Controller Fatal Status.** Once set to 1, the controller has experienced an unrecoverable fatal condition; host must reset it.

---

### Test 4.18 – Version Register (VS)

**Q92. [E]** What are the field layouts of the VS register?
**A.** MJR[31:16], MNR[15:8], TER[7:0] (e.g. VS = `0x00020000` → NVMe 2.0).

---

### Test 4.20 – CRIMT

**Q93. [H]** What does CRIMT (Controller Ready Independent of Media Timeout) describe?
**A.** Maximum time (100 ms units) for CSTS.RDY to become 1 when CC.CRIME = 1, indicating the controller is ready to accept admin commands independent of media being ready.

---

## Group 5 – System Memory Structure

### Test 5.1 – PRP Base Address and Offset (PBAO)

**Q94. [M]** A PRP entry's offset is not aligned to the controller's Memory Page Size (CC.MPS). What happens?
**A.** Command aborted with **PRP Offset Invalid (`13h`)**.

**Q95. [H]** Under what conditions may PRP Entry 2 be 0?
**A.** When the entire transfer fits in a single page starting at PRP Entry 1 (≤ one MPS in size). Otherwise PRP Entry 2 must either contain the second data buffer's PRP or a PRP List.

---

### Test 5.3 / 5.4 – Status Field / Generic Command Status

**Q96. [E]** Where in the CQE is the Status Code Type (SCT)?
**A.** **Dword 3, bits 27:25.**

**Q97. [E]** What are the three main Status Code Types?
**A.** **`0h` Generic, `1h` Command Specific, `2h` Media and Data Integrity Errors** (plus `7h` Vendor Specific and others).

**Q98. [M]** Status code `0Bh` in the Generic group — what is it?
**A.** **Invalid Namespace or Format.**

**Q99. [M]** Status `81h` in the Media and Data Integrity group?
**A.** **Unrecovered Read Error.**

**Q100. [H]** What is the **M (More)** bit in the Status Field, and what is its consequence?
**A.** M=1 indicates additional error information for this command is available in the **Error Information log page (LID 01h)**. After M=1, a subsequent Get Log Page to LID 01h must include a new entry for that error.

---

## Group 6 – Controller Architecture

### Test 6.1 – Controller Level Reset

**Q101. [M]** Three controller-level reset mechanisms — name them.
**A.** **Controller Reset** (CC.EN 1→0), **NVM Subsystem Reset** (NSSR.NSSRC = `4E564D65h` = "NVMe"), **PCIe Function-Level Reset / Conventional Reset**.

**Q102. [H]** What state survives a Controller Reset that does **not** survive an NVM Subsystem Reset?
**A.** Persistent host-controller associations (Host Identifier, persistent reservations with PTPL=0 cleared on subsystem reset only when PTPL=0, etc.) — specifically Sanitize Media Verification State exits on subsystem reset but not necessarily on controller reset (Test 1.17 Case 13).

---

## Group 7 – Reservations

### Test 7.1 – Reservation Report

**Q103. [E]** Which ONCS bit advertises Reservations support?
**A.** ONCS **bit 5**.

**Q104. [M]** A namespace with no registrants is reported by Reservation Report. What does the Reservation Status data structure show?
**A.** Number of Registered Controllers (NUMCTRL) = 0; the Reservation Type (RTYPE) field is `00h` (No reservation); the registered controllers list is empty.

---

### Test 7.2 – Reservation Registration

**Q105. [E]** Which RREGA value registers a new key?
**A.** `000b` — **Register Reservation Key**, supplying NRKEY.

**Q106. [M]** A host already registered tries to register again with a **different** key (without IEKEY=1). Result?
**A.** **Reservation Conflict.** The previously registered key is unchanged.

**Q107. [H]** A host wants to replace its key without supplying the current key. What's the mechanism?
**A.** Set **IEKEY=1** in the Reservation Register command; the controller skips CRKEY verification.

**Q108. [H]** What does the `CPTPL` (Change Persist Through Power Loss) field encoding `11b` do?
**A.** Sets **PTPL state to 1** — reservation state persists across power cycles. `10b` = clear PTPL to 0. `00b/01b` = no change.

---

### Test 7.4 – Acquiring a Reservation

**Q109. [E]** Which RACQA value acquires a reservation?
**A.** `000b` — Acquire.

**Q110. [M]** Reservation Acquire is sent with `CRKEY` that does not match the registered key. Status?
**A.** **Reservation Conflict.**

---

### Test 7.6 – Preempting

**Q111. [M]** Which RACQA value performs a Preempt (without abort)? Preempt and Abort?
**A.** `001b` = Preempt, `010b` = Preempt and Abort.

---

### Test 7.8 – Command Behavior with Different Reservation Types

**Q112. [H]** A namespace has a `Write Exclusive – Registrants Only` (RTYPE `05h`) reservation. A non-registered host issues a Write. Result?
**A.** **Reservation Conflict.** Reads from non-registrants are allowed; Writes are not.

**Q113. [H]** With `Exclusive Access – All Registrants` (RTYPE `06h`), what commands may a non-reservation-holding registrant execute?
**A.** None of read/write — all I/O is denied for non-holders; only the reservation holder has access. Non-registrants are also denied.

---

## Group 8 – Namespace Management

### Test 8.2 – Namespace Management Command

**Q114. [E]** Which OACS bit advertises Namespace Management support?
**A.** OACS **bit 3**.

**Q115. [M]** What value of the `SEL` field in Namespace Management means "Create"? "Delete"?
**A.** SEL `0h` = Create, SEL `1h` = Delete.

**Q116. [H]** After a successful Create Namespace, the new namespace is allocated but not attached. Which command must follow before host I/O can target it?
**A.** **Namespace Attachment** (opcode `15h`) with SEL=0 (Controller Attach), specifying the controller IDs.

---

## Group 9 – Flexible Data Placement (FDP)

### Test 9.1 – FDP Configuration Log

**Q117. [E]** Which Log Identifier returns the FDP Configuration?
**A.** **LID `20h`.**

**Q118. [E]** Which LID returns Reclaim Unit Handle Usage? FDP Statistics? FDP Events?
**A.** **`21h`** Reclaim Unit Handle Usage, **`22h`** Statistics, **`23h`** Events.

**Q119. [M]** Which feature identifier enables FDP? Which exposes FDP Events configuration?
**A.** FID **`1Dh`** Enable Flexible Data, FID **`1Eh`** FDP Events.

---

## Group 10 – Reachability Groups

**Q120. [M]** Reachability Groups are reported via what log? What is the new asynchronous event they trigger?
**A.** Reachability Groups log; AEN type associated with namespace attribute change (NSPACE attribute change AEN per Test 10.1).

---

## Group 11 – Boot Partition

### Test 11.1 – Boot Partition Write Protection

**Q121. [E]** Where is the Boot Partition size reported?
**A.** In **Identify Controller** — the **BPS** field (Boot Partition Size).

**Q122. [M]** What is the LID for the Boot Partition log? What feature identifier locks the BP from writes?
**A.** **LID `15h`** Boot Partition log. FID `15h` Boot Partition Write Protection Configuration (per spec; verified by Test 11.1).

---

## Group 12 – Host Managed Live Migration

### Test 12.1 / 12.2 – Migration Receive / Send

**Q123. [E]** What feature distinguishes the "Migration Receive" command from "Migration Send"?
**A.** **Direction**: Receive pulls controller state from the source DUT into host memory; Send pushes controller state from the host into the destination DUT.

**Q124. [M]** Which Identify Controller field indicates Host-Managed Live Migration support?
**A.** The **HMLMS** (Host Managed Live Migration Support) bit in the Identify Controller data structure (verified by Test 12.1 Case 1).

**Q125. [H]** During a "Track Send" the controller is expected to begin recording dirty pages. What identifier in the command selects which tracking is enabled?
**A.** **Track Action** within the Migration Track Send command — values select Start Tracking, Stop Tracking, Get Dirty Page Log, etc. (Test 12.5).

---

## Cross-cutting / Concept Questions

**Q126. [E]** What does **(M, OF)** vs **(M, OF-FYI)** mean in a test header?
**A.** First label = PCIe/local applicability, second = Fabrics applicability. **M** = Mandatory, **FYI** = For Your Information (informational only), **IP** = In Progress. So `(M, OF-FYI)` means mandatory for local NVMe but informational for fabrics.

**Q127. [E]** What does "DUT" stand for in the test plan?
**A.** **Device Under Test.**

**Q128. [M]** Why does the test plan repeatedly say "Verify that all received responses have all Reserved fields set to 0"?
**A.** The spec mandates that reserved fields in CQEs be cleared to 0 by the controller. Non-zero reserved fields can break forward compatibility when those bits are later defined and are a common conformance failure.

**Q129. [M]** A test specifies `(M, OF)` and your DUT is a fabrics target. Is the test applicable?
**A.** **Yes** — if the OF label is `OF` (not `OF-FYI` or absent), the test is mandatory for fabrics. `OF-FYI` would make it informational; no OF label would make it PCIe-only.

**Q130. [H]** During benchwork, why is verifying behavior with `NSID = FFFFFFFFh` (broadcast NSID) so important across so many tests?
**A.** Broadcast NSID behavior differs by spec version and command — some commands accept it (Identify CNS=02h, Get Log Page, Format NVM, Sanitize, DST), some explicitly reject it on 1.4+ (Dataset Management, Read/Write/Compare), and some only allow it for specific feature/log combinations. Mishandling it is a common silent conformance gap.

**Q131. [H]** If two tests describe the same scenario but one is labeled `(M)` and another is labeled `(IP)`, which one do you run for a Conformance log submission?
**A.** Only the **M** (Mandatory) variant counts toward the conformance pass/fail. **IP** (In Progress) tests are still under development and are not yet part of the official conformance scoring.

---

## Quick Reference – Status Codes You Should Memorize

| Code | Group | Name |
|------|-------|------|
| `00h` | Generic | Successful Completion |
| `01h` | Generic | Invalid Command Opcode |
| `02h` | Generic | Invalid Field in Command |
| `05h` | Generic | Data Transfer Error |
| `07h` | Generic | Command Abort Requested |
| `09h` | Generic | Invalid Log Page |
| `0Bh` | Generic | Invalid Namespace or Format |
| `0Eh` | Generic | Command Sequence Error |
| `13h` | Generic | PRP Offset Invalid |
| `1Dh` | Generic | Sanitize In Progress |
| `80h` | Generic | LBA Out of Range |
| `00h` | Cmd-Spec | Completion Queue Invalid |
| `01h` | Cmd-Spec | Invalid Queue Identifier |
| `02h` | Cmd-Spec | Invalid Queue Size |
| `03h` | Cmd-Spec | Abort Command Limit Exceeded |
| `05h` | Cmd-Spec | Asynchronous Event Request Limit Exceeded |
| `08h` | Cmd-Spec | Invalid Interrupt Vector |
| `0Ch` | Cmd-Spec | Invalid Queue Deletion |
| `83h` | Cmd-Spec | Reservation Conflict |
| `80h` | Media | Write Fault |
| `81h` | Media | Unrecovered Read Error |
| `82h` | Media | End-to-end Guard Check Error |
| `85h` | Media | Compare Failure |
| `86h` | Media | Access Denied |

---

## Studying Tips for the Bench

1. Always start by **reading the Identify Controller and Identify Namespace data structures** — every test depends on capability bits in there.
2. **OACS, ONCS, FNA, SANICAP, NWPC, OAES** are the most-referenced capability fields. Memorize their bit layouts.
3. For every "is X supported?" question, the path is: Identify → check capability bit → conditionally run test → log "N/A" if unsupported. Never assume.
4. **Reserved fields = 0** and **CQE ordering** (aborted command's CQE before Abort's CQE) catch a huge fraction of real conformance bugs.
5. When polling background ops (Sanitize, DST, Format with FNA bit set), the **Get Log Page** for that operation's log is what you trust — not just the command completion.
