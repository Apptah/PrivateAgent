import SwiftUI

struct InputBar: View {
    @Bindable var viewModel: ChatViewModel
    @FocusState private var isFocused: Bool

    var body: some View {
        HStack(alignment: .bottom, spacing: 10) {
            TextField("Message...", text: $viewModel.inputText, axis: .vertical)
                .lineLimit(1...5)
                .padding(.horizontal, 14)
                .padding(.vertical, 10)
                .background(.quaternary, in: RoundedRectangle(cornerRadius: 20))
                .focused($isFocused)
                .disabled(viewModel.isGenerating)
                .onSubmit {
                    if !viewModel.isGenerating && !viewModel.inputText.isEmpty {
                        isFocused = false
                        viewModel.sendMessage()
                    }
                }

            if viewModel.isGenerating {
                Button {
                    viewModel.cancel()
                } label: {
                    Image(systemName: "stop.circle.fill")
                        .resizable()
                        .frame(width: 32, height: 32)
                        .foregroundStyle(.red)
                }
                .buttonStyle(.plain)
            } else {
                Button {
                    isFocused = false
                    viewModel.sendMessage()
                } label: {
                    Image(systemName: "arrow.up.circle.fill")
                        .resizable()
                        .frame(width: 32, height: 32)
                        .foregroundStyle(viewModel.inputText.isEmpty ? Color.secondary : Color.accentColor)
                }
                .buttonStyle(.plain)
                .disabled(viewModel.inputText.isEmpty)
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 10)
        .background(.bar)
    }
}
