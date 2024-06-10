import sys
from correction import spell_correction
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLineEdit, QTextEdit, QPushButton

class SpellCorrector(QWidget):
    def __init__(self):
        super().__init__()

        self.corrector = spell_correction()
        self.initUI()

    def initUI(self):
        # 创建布局
        layout = QVBoxLayout()

        # 创建输入框
        self.input_box = QLineEdit(self)
        self.input_box.setPlaceholderText("Enter text here...")
        layout.addWidget(self.input_box)

        # 创建输出框
        self.output_box = QTextEdit(self)
        self.output_box.setReadOnly(True)
        layout.addWidget(self.output_box)

        # 创建按钮
        self.button = QPushButton("Correct Spelling", self)
        self.button.clicked.connect(self.correct_spelling)
        layout.addWidget(self.button)

        # 设置布局
        self.setLayout(layout)

        # 设置窗口属性
        self.setWindowTitle('Spell Corrector')
        self.show()

    def correct_spelling(self):
        # 获取输入框的文本
        input_text = self.input_box.text()

        # 进行拼写校正
        corrected_text = self.spell_correct(input_text)

        # 显示校正后的文本
        self.output_box.setText(corrected_text)

    def spell_correct(self, text):
        # 这里使用之前的拼写校正逻辑来实现文本的校正
        # 这是一个示例，实际应用中需要使用你具体的拼写校正函数
        corrected_sentence = self.corrector.correction(text)
        return corrected_sentence


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = SpellCorrector()
    sys.exit(app.exec_())
