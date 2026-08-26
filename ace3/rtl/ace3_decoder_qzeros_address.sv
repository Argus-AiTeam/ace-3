`default_nettype none

module ace3_decoder_qzeros_address (
    input  wire        meta_live_i,
    input  wire [2:0]  projection_kind_i,
    input  wire [5:0]  group_i,
    input  wire [9:0]  word_i,
    output reg         address_valid_o,
    output reg  [15:0] address_o
);
    always @* begin
        address_valid_o = 1'b0;
        address_o = 16'd0;
        if (meta_live_i) begin
            case (projection_kind_i)
              3'd0, 3'd3: begin
                  if (group_i < 7 && word_i < 112) begin
                      address_o = group_i * 16'd112 + {6'd0,word_i};
                      address_valid_o = 1'b1;
                  end
              end
              3'd1, 3'd2: begin
                  if (group_i < 7 && word_i < 16) begin
                      address_o = group_i * 16'd16 + {6'd0,word_i};
                      address_valid_o = 1'b1;
                  end
              end
              3'd4, 3'd5: begin
                  if (group_i < 7 && word_i < 608) begin
                      address_o = group_i * 16'd608 + {6'd0,word_i};
                      address_valid_o = 1'b1;
                  end
              end
              3'd6: begin
                  if (group_i < 38 && word_i < 112) begin
                      address_o = group_i * 16'd112 + {6'd0,word_i};
                      address_valid_o = 1'b1;
                  end
              end
              default: begin
                  address_valid_o = 1'b0;
                  address_o = 16'd0;
              end
            endcase
        end
    end
endmodule

`default_nettype wire
