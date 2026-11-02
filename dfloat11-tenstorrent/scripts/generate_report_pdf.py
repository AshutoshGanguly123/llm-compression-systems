"""Generate a PDF midterm-style progress report for the DFloat11 on Tenstorrent project."""

from fpdf import FPDF


class ReportPDF(FPDF):
    def __init__(self, orientation="P", unit="mm", format="A4"):
        super().__init__(orientation=orientation, unit=unit, format=format)
        self.alias_nb_pages()

    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("Helvetica", "B", 8)
        self.set_text_color(100, 100, 100)
        self.cell(0, 4, "DFloat11 Lossless Weight Compression on Tenstorrent", align="L")
        self.cell(0, 4, "April 2026", align="R", new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(180, 180, 180)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(2)

    def footer(self):
        self.set_y(-10)
        self.set_font("Helvetica", "I", 7)
        self.set_text_color(150, 150, 150)
        self.cell(0, 8, f"Page {self.page_no()}/{{nb}}", align="C")

    def section(self, title):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(20, 55, 110)
        self.cell(0, 5.5, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(0.5)

    def subsection(self, title):
        self.set_font("Helvetica", "B", 8.5)
        self.set_text_color(50, 50, 50)
        self.cell(0, 5, title, new_x="LMARGIN", new_y="NEXT")

    def para(self, text):
        self.set_font("Helvetica", "", 8.5)
        self.set_text_color(30, 30, 30)
        self.multi_cell(0, 4, text)
        self.ln(0.5)

    def bullet(self, text):
        self.set_font("Helvetica", "", 8.5)
        self.set_text_color(30, 30, 30)
        x0 = self.l_margin
        self.set_x(x0 + 3)
        self.multi_cell(self.w - self.l_margin - self.r_margin - 3, 4, "- " + text)

    def add_table(self, headers, rows, col_widths=None, hdr_size=7.5, row_size=7.5, row_h=4.5):
        if col_widths is None:
            col_widths = [190 / len(headers)] * len(headers)
        self.set_font("Helvetica", "B", hdr_size)
        self.set_fill_color(30, 65, 120)
        self.set_text_color(255, 255, 255)
        for i, h in enumerate(headers):
            self.cell(col_widths[i], row_h + 0.5, h, border=1, fill=True, align="C")
        self.ln()
        self.set_font("Helvetica", "", row_size)
        self.set_text_color(30, 30, 30)
        alt = False
        for row in rows:
            self.set_fill_color(242, 246, 252) if alt else self.set_fill_color(255, 255, 255)
            for i, c in enumerate(row):
                a = "L" if i == 0 else "C"
                self.cell(col_widths[i], row_h, str(c), border=1, fill=True, align=a)
            self.ln()
            alt = not alt
        self.ln(1.5)


def main():
    pdf = ReportPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=12)

    # ===== PAGE 1 =====
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 15)
    pdf.set_text_color(20, 45, 95)
    pdf.cell(0, 8, "DFloat11: Lossless BF16 Weight Compression", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(70, 70, 70)
    pdf.cell(0, 6, "with On-Device Huffman Decode on Tenstorrent Blackhole", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(90, 90, 90)
    pdf.cell(0, 5, "Midterm progress report", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_draw_color(30, 65, 120)
    pdf.set_line_width(0.5)
    pdf.line(30, pdf.get_y() + 1, 180, pdf.get_y() + 1)
    pdf.ln(4)

    pdf.section("1. Problem & Approach")
    pdf.para(
        "Large language models need a lot of weight storage. Getting those weights from the host "
        "machine onto the accelerator quickly matters for load time and for how much compressed "
        "storage you can use in practice. DFloat11 uses a simple observation about BF16 weights: "
        "each value is 16 bits (sign, exponent, mantissa). In trained models, exponents repeat a "
        "small set of values far more often than random data would. We Huffman-encode the exponent "
        "stream, pack sign and mantissa compactly, and reconstruct weights with no loss. The "
        "question we focused on is whether decoding can run on the accelerator itself, so the "
        "host mainly moves compressed bytes and the chip expands them where the weights are needed."
    )

    pdf.section("2. Key Result: Bit-Exact On-Device Decode")
    pdf.para(
        "We implemented a TT-Metalium data-movement kernel on a single Tensix RISC-V core on a "
        "Tenstorrent Blackhole chip. The kernel reads compressed data from on-chip DRAM, performs "
        "table-based Huffman decode in local SRAM, rebuilds BF16 values, and writes the result "
        "back to device memory. The host does not participate in the decode loop. We verified "
        "bit-exact agreement with a CPU reference at full tensor scale (zero mismatches)."
    )

    pdf.subsection("Experimental Results (Llama-3.2-1B-Instruct)")
    pdf.add_table(
        ["Metric", "Test Layer (2048x128)", "lm_head.weight (128256x2048)"],
        [
            ["Elements", "262,144", "262,668,288"],
            ["Original / Compressed", "0.50 / 0.33 MB", "501 / 332 MB"],
            ["Compression ratio", "1.51x (33.8% saved)", "1.51x (33.7% saved)"],
            ["Bit-exact match", "YES (0 mismatches)", "YES (0 mismatches)"],
            ["CPU decode time", "19.6 ms (25.5 MB/s)", "16,641 ms (30.1 MB/s)"],
            ["Device decode (1 core)", "65 ms (7.7 MB/s)", "47,709 ms (10.5 MB/s)"],
            ["DRAM upload / readback", "1.0 / 0.7 ms", "316 / 366 ms"],
        ],
        col_widths=[52, 69, 69],
    )

    pdf.section("3. System Architecture")
    pdf.para(
        "Offline, weights are compressed anywhere and saved as artifacts (Huffman bitstream, packed "
        "sign/mantissa, codebook). At runtime, the host sends only the compressed form to device "
        "memory (about a third less data than raw BF16 for this layer), then launches the Tensix "
        "kernel to decompress in place. The decoded BF16 weights sit in device DRAM and are ready "
        "for normal matrix operations."
    )
    pdf.set_font("Courier", "", 7)
    pdf.set_text_color(40, 40, 40)
    pdf.multi_cell(
        0,
        3.2,
        "  [Host storage] --> upload --> [Device memory: compressed] --> read --> [Local SRAM]\n"
        "                    (~332 MB)     bitstream + sign/mantissa              Huffman decode\n"
        "                                                                 rebuild BF16\n"
        "                    [Device memory: BF16 output]  <-- write <-- [Local SRAM]\n"
        "                     (~501 MB, ready for matmul)",
    )
    pdf.ln(2)

    pdf.section("4. Kernel Design: Table-Based Huffman Decode")
    pdf.para(
        "A naive Huffman decoder walks a tree one bit at a time, which is slow on a small embedded "
        "core. Instead we use a flat lookup table (2^L entries for max code length L; 13 bits gives "
        "a 16 KB table). Each entry stores the symbol and length. Decode becomes: peek bits, index "
        "the table, emit, advance. The kernel streams 32K-element chunks from DRAM through SRAM and "
        "writes decoded BF16 back."
    )

    pdf.subsection("L1 Memory Budget (per 32K-element chunk)")
    pdf.add_table(
        ["Region", "Size", "Purpose"],
        [
            ["Decode LUT", "16 KB", "2^13 entries x 2 bytes"],
            ["Bitstream chunk", "~85 KB", "Compressed exponent data"],
            ["Sign/mantissa", "32 KB", "1 byte/element, passthrough"],
            ["Output BF16", "64 KB", "Decoded 16-bit values"],
            ["Total", "~197 KB", "14% of 1.4 MB Tensix L1"],
        ],
        col_widths=[40, 28, 122],
    )

    # ===== PAGE 2 =====
    pdf.add_page()

    pdf.section("5. Why It Works: Exponent Distribution")
    pdf.para(
        "The Llama-3.2-1B lm_head.weight has 262M BF16 elements but only 19 unique exponent "
        "values. The top four cover most of the mass and receive short Huffman codes, giving "
        "about 3x compression on the exponent stream alone."
    )

    pdf.add_table(
        ["Exponent", "Huffman Code", "Bits", "Count", "Share"],
        [
            ["121", "11", "2", "83,128,159", "31.6%"],
            ["120", "01", "2", "68,497,745", "26.1%"],
            ["119", "101", "3", "39,099,673", "14.9%"],
            ["122", "100", "3", "30,672,859", "11.7%"],
            ["118", "000", "3", "20,154,937", "7.7%"],
            ["117", "0010", "4", "10,441,946", "4.0%"],
            ["116", "00110", "5", "5,024,632", "1.9%"],
            ["Other (12)", "7-13 bits", "-", "5,648,337", "2.1%"],
        ],
        col_widths=[25, 35, 15, 45, 22],
        hdr_size=7,
        row_size=7,
        row_h=4,
    )

    pdf.section("6. Scaling Path to Higher Throughput")
    pdf.para(
        "The current proof uses one core and reaches about 10.5 MB/s on the large tensor. Blackhole "
        "has on the order of 140 Tensix cores, and this style of decode is embarrassingly parallel "
        "at chunk granularity."
    )

    pdf.add_table(
        ["Cores", "Est. Throughput", "lm_head Decode Time", "Notes"],
        [
            ["1 (current)", "10.5 MB/s", "47.7 s", "Single RISC-V proof"],
            ["10", "~105 MB/s", "~4.8 s", "Straightforward parallelization"],
            ["70", "~735 MB/s", "~0.68 s", "Half-chip utilization (illustrative)"],
            ["140", "~1.4 GB/s", "~0.34 s", "Full chip; host transfer often limits first"],
        ],
        col_widths=[28, 38, 45, 79],
    )

    pdf.subsection("Further Optimizations")
    pdf.bullet("Wider bitstream reads (e.g. 32-bit words) instead of fine-grained peeks")
    pdf.bullet("Overlapping DMA with decode (double buffering)")
    pdf.bullet("Using both data movers per Tensix where available")
    pdf.bullet("Fusing decode with the consumer kernel when possible")
    pdf.ln(1)

    pdf.section("7. End-to-End Impact (Illustrative)")
    pdf.para(
        "For a full Llama-3.2-1B model (~2.5 GB BF16 weights), moving uncompressed weights from host "
        "to device at 25 GB/s is on the order of 100 ms. With 1.51x compression, less data crosses "
        "the link (~66 ms at the same effective rate). If decode is spread across many cores, the "
        "decode time can drop to a few milliseconds in a rough projection. For larger models the "
        "same idea applies: smaller artifacts on disk and less time moving bytes before inference."
    )

    pdf.section("8. Conclusions (work to date)")
    pdf.bullet(
        "Lossless compression at about 1.51x on a real lm_head, verified bit-exact on 262M elements."
    )
    pdf.bullet("On-device Huffman decode on Tensix, without the host in the inner decode loop.")
    pdf.bullet(
        "Smaller uploads from host for compressed weights; decoded tensors ready for normal compute."
    )
    pdf.bullet(
        "Clear path to raise throughput by parallelizing across many cores; host-device transfer "
        "often becomes the next bottleneck before decode does."
    )

    pdf.ln(2)
    pdf.set_draw_color(180, 180, 180)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(2)
    pdf.set_font("Helvetica", "I", 7.5)
    pdf.set_text_color(110, 110, 110)
    pdf.cell(
        0,
        3.5,
        "Hardware: Tenstorrent Blackhole system (multi-chip box), tt-metal SDK, Llama-3.2-1B-Instruct",
        align="C",
        new_x="LMARGIN",
        new_y="NEXT",
    )
    pdf.cell(
        0,
        3.5,
        "Implementation: Python offline compression + C++ TT-Metalium kernel + verification pipeline",
        align="C",
        new_x="LMARGIN",
        new_y="NEXT",
    )

    # ===== PAGE 3 =====
    pdf.add_page()
    pdf.section("9. Major Future Work")
    pdf.para(
        "The results above establish lossless compression and on-accelerator decode for representative "
        "layers. The main story we want to demonstrate next is a full language model running on "
        "Tenstorrent with DFloat11 in the loop - not only microbenchmarks, but an end-to-end path that "
        "a reader can recognize as real inference."
    )
    pdf.subsection("Target by 30 April 2026")
    pdf.bullet(
        "End-to-end LLM on Tenstorrent: run a supported open model through tt-metal / tt_transformers "
        "with coherent text generation (vanilla weights first, then integrate DFloat11 at the lm_head "
        "or another high-value linear layer as the clearest paper-aligned demo)."
    )
    pdf.bullet(
        "Show that compressed artifacts can sit on the critical path of that stack (load, decode, "
        "match reference outputs within the chosen tolerance), without changing model behavior beyond "
        "what lossless compression allows."
    )
    pdf.bullet(
        "Document the demo with a short reproducible recipe (environment, config, one command or "
        "script) suitable for reviewers or partners."
    )
    pdf.ln(1)
    pdf.subsection("Additional directions (after the April milestone)")
    pdf.bullet("Scale device-side decode across many Tensix cores and measure end-to-end load time")
    pdf.bullet("Explore host-side decode optimizations or hybrid pipelines where helpful")
    pdf.bullet("Extend artifact tooling to more layers and model families used internally")
    pdf.bullet("Tighter integration with serving or batching stacks if product asks for it")
    pdf.ln(3)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(80, 80, 80)
    pdf.multi_cell(
        0,
        4,
        "This report summarizes progress to the midterm checkpoint. The April 30 goal centers on "
        "delivering the full LLM-on-TT demonstration with DFloat11 as the headline compression story.",
    )

    out_path = "/home/ashutoshganguly/foais_project_dfloat11/outputs/DFloat11_Device_Decode_Report.pdf"
    pdf.output(out_path)
    print(f"PDF saved to: {out_path}")


if __name__ == "__main__":
    main()
